import psutil
import yaml
import argparse
import torch
import random
import os
import wandb
import pickle
import numpy as np
import json
import torch.nn.functional as F
import time
import soundfile as sf

from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
from accelerate import Accelerator
from math import isclose

from AutoMasher.fyp import Audio
from config import TrainingConfig, STFT, make_stft
from model import (
    Vocoder,
    Discriminator,
    DiscriminatorOutput,
    Vggish
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training splits
TRAIN_SPLIT_PERCENTAGE = 0.8
VALIDATION_SPLIT_PERCENTAGE = 0.1
TEST_SPLIT_PERCENTAGE = 0.1

assert isclose(TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE, 1.0)

ALLOWED_EXTENSIONS = {".wav", ".mp3"}

WAV_READ_BUFFER = 1024


def _assert(condition: bool, message: str = ""):
    # Uncomment this line to disable all checking
    # return
    if not condition:
        raise AssertionError(message)


class COMP5421Dataset(torch.utils.data.Dataset):
    def __init__(self, config: TrainingConfig, files: list[str]):
        # Find all the audio files in the dataset directory
        self._files = files
        self.config = config
        self.stft = STFT(config.nfft, config.ntimeframes)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int):
        path = self._files[idx]
        # If it is an wav file on a separate disk, read only the slice to save disk i/o
        audios: list[Audio]
        if is_external_drive(path) and path.endswith(".wav"):
            audios = read_random_audio_slice(path, self.config.audio_length, self.config.sample_rate, self.config.ds_batch_size)
        else:
            audios = []
            audio = Audio.load(path).resample(self.config.sample_rate)
            if audio.nframes - self.config.audio_length <= 0:
                return self.__getitem__(np.random.randint(0, len(self._files)))
            for i in range(self.config.ds_batch_size):
                slice_start = np.random.randint(0, audio.nframes - self.config.audio_length)
                aud = audio.slice_frames(slice_start, slice_start + self.config.audio_length)
                audios.append(aud)
        audios = [x.to_nchannels(1) for x in audios]
        for audio in audios:
            _assert(audio.sample_rate == self.config.sample_rate, f"Audio sample rate {audio.sample_rate} does not match config sample rate {self.config.sample_rate}.")
            _assert(audio.nframes == self.config.audio_length, f"Audio length {audio.nframes} does not match config audio length {self.config.audio_length}.")
        return torch.stack([x._data[0].float() for x in audios])


class DiscriminatorLoss(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

    def compute_bce_loss(self, dgz: DiscriminatorOutput, dx: DiscriminatorOutput | None = None):
        generator = dx is None
        if generator:
            loss = self.config.disc_spec_weight * F.binary_cross_entropy_with_logits(
                dgz.spectrogram_results, torch.ones_like(dgz.spectrogram_results)
            )
            loss_audio = torch.zeros_like(loss)
            for i in range(self.config.num_audio_discriminators):
                loss_audio += self.config.disc_audio_weights[i] * F.binary_cross_entropy_with_logits(
                    dgz.audio_results[i], torch.ones_like(dgz.audio_results[i])
                )
        else:
            fake_loss = self.config.disc_spec_weight * F.binary_cross_entropy_with_logits(
                dgz.spectrogram_results, torch.zeros_like(dgz.spectrogram_results)
            )
            real_loss = self.config.disc_spec_weight * F.binary_cross_entropy_with_logits(
                dx.spectrogram_results, torch.ones_like(dx.spectrogram_results)
            )
            fake_loss_audio = torch.zeros_like(fake_loss)
            real_loss_audio = torch.zeros_like(real_loss)
            for i in range(self.config.num_audio_discriminators):
                fake_loss_audio += self.config.disc_audio_weights[i] * F.binary_cross_entropy_with_logits(
                    dgz.audio_results[i], torch.zeros_like(dgz.audio_results[i])
                )
                real_loss_audio += self.config.disc_audio_weights[i] * F.binary_cross_entropy_with_logits(
                    dx.audio_results[i], torch.ones_like(dx.audio_results[i])
                )
            loss = (fake_loss + real_loss) / 2.0
            loss_audio = (fake_loss_audio + real_loss_audio) / 2.0
        return loss, loss_audio

    def compute_mse_loss(self, dgz: DiscriminatorOutput, dx: DiscriminatorOutput | None = None):
        generator = dx is None
        if generator:
            loss = self.config.disc_spec_weight * F.mse_loss(
                dgz.spectrogram_results, torch.ones_like(dgz.spectrogram_results)
            )
            loss_audio = torch.zeros_like(loss)
            for i in range(self.config.num_audio_discriminators):
                loss_audio += self.config.disc_audio_weights[i] * F.mse_loss(
                    dgz.audio_results[i], torch.ones_like(dgz.audio_results[i])
                )
        else:
            fake_loss = self.config.disc_spec_weight * F.mse_loss(
                dgz.spectrogram_results, torch.zeros_like(dgz.spectrogram_results)
            )
            real_loss = self.config.disc_spec_weight * F.mse_loss(
                dx.spectrogram_results, torch.ones_like(dx.spectrogram_results)
            )
            fake_loss_audio = torch.zeros_like(fake_loss)
            real_loss_audio = torch.zeros_like(real_loss)
            for i in range(self.config.num_audio_discriminators):
                fake_loss_audio += self.config.disc_audio_weights[i] * F.mse_loss(
                    dgz.audio_results[i], torch.zeros_like(dgz.audio_results[i])
                )
                real_loss_audio += self.config.disc_audio_weights[i] * F.mse_loss(
                    dx.audio_results[i], torch.ones_like(dx.audio_results[i])
                )
            loss = (fake_loss + real_loss) / 2.0
            loss_audio = (fake_loss_audio + real_loss_audio) / 2.0
        return loss, loss_audio

    def compute_hinge_loss(self, dgz: DiscriminatorOutput, dx: DiscriminatorOutput | None = None):
        generator = dx is None
        if generator:
            loss = self.config.disc_spec_weight * F.relu(1 - dgz.spectrogram_results).mean()
            loss_audio = torch.zeros_like(loss)
            for i in range(self.config.num_audio_discriminators):
                loss_audio += self.config.disc_audio_weights[i] * F.relu(1 - dgz.audio_results[i]).mean()
        else:
            fake_loss = self.config.disc_spec_weight * F.relu(1 + dgz.spectrogram_results).mean()
            real_loss = self.config.disc_spec_weight * F.relu(1 - dx.spectrogram_results).mean()
            fake_loss_audio = torch.zeros_like(fake_loss)
            real_loss_audio = torch.zeros_like(real_loss)
            for i in range(self.config.num_audio_discriminators):
                fake_loss_audio += self.config.disc_audio_weights[i] * F.relu(1 + dgz.audio_results[i]).mean()
                real_loss_audio += self.config.disc_audio_weights[i] * F.relu(1 - dx.audio_results[i]).mean()
            loss = (fake_loss + real_loss) / 2.0
            loss_audio = (fake_loss_audio + real_loss_audio) / 2.0
        return loss, loss_audio

    def forward(self, dgz: DiscriminatorOutput, dx: DiscriminatorOutput | None = None):
        loss = loss_audio = None
        if self.config.disc_loss == "bce":
            loss, loss_audio = self.compute_bce_loss(dgz, dx)
        elif self.config.disc_loss == "mse":
            loss, loss_audio = self.compute_mse_loss(dgz, dx)
        elif self.config.disc_loss == "hinge":
            loss, loss_audio = self.compute_hinge_loss(dgz, dx)
        else:
            raise ValueError(f"Unsupported discriminator loss type: {self.config.disc_loss}")
        loss_audio = loss_audio / self.config.num_audio_discriminators
        return loss, loss_audio


def is_external_drive(path: str) -> bool:
    """Returns True if the path is on an external drive. Might not be 100% accurate."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return False  # Path does not exist
    if os.name == 'nt':
        return os.path.splitdrive(path)[0] != "C:"
    partitions = psutil.disk_partitions()
    path_mount_point = None

    # Find the mount point for the path
    for part in partitions:
        if path.startswith(part.mountpoint):
            if path_mount_point is None or len(part.mountpoint) > len(path_mount_point.mountpoint):
                path_mount_point = part

    if path_mount_point is None:
        return False  # Path mount point not found

    # On Windows, 'opts' might include 'removable' for external drives
    # On Unix-like systems, 'device' might point to an external device path
    # You might need additional checks based on the OS and filesystem specifics
    return 'removable' in path_mount_point.opts


def load_lpips(config: TrainingConfig) -> nn.Module:
    class _PerceptualLossWrapper(nn.Module):
        def __init__(self, in_sr: int):
            super().__init__()
            self.lpips = Vggish()
            self.in_sr = in_sr

        def forward(self, pred_audio, targ_audio):
            pred1 = self.lpips((pred_audio, self.in_sr))
            targ1 = self.lpips((targ_audio, self.in_sr))
            pred2 = self.lpips((pred_audio, self.lpips.input_sr))
            targ2 = self.lpips((targ_audio, self.lpips.input_sr))
            return (F.mse_loss(pred1, targ1) + F.mse_loss(pred2, targ2)) * 0.5

    model = _PerceptualLossWrapper(config.sample_rate)
    model = model.eval().to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def find_all_audio_files(config: TrainingConfig) -> list[str]:
    files: list[str] = []
    for dirs in config.dataset_dirs:
        for file in os.listdir(dirs):
            if os.path.splitext(file)[1].lower() in ALLOWED_EXTENSIONS:
                files.append(os.path.join(dirs, file))
    return files


def make_split(config: TrainingConfig) -> tuple[list[str], list[str], list[str]]:
    """
    Splits the dataset into train, validation, and test sets.

    Args:
    config (TrainingConfig): Configuration object containing the dataset directories and other parameters.
    paths (list[str]): List of file paths to the audio files.
    """
    # If the dataset is already split before, load the split from the file
    if os.path.exists(config.dataset_split_path):
        with open(config.dataset_split_path, "rb") as f:
            dataset_split = pickle.load(f)
            past_train_files = dataset_split["train"]
            past_val_files = dataset_split["val"]
            past_test_files = dataset_split["test"]
    else:
        past_train_files = []
        past_val_files = []
        past_test_files = []

    randomizer = random.Random(config.seed)

    # Find all the audio files in the dataset directory
    paths = find_all_audio_files(config)
    train_split = []
    val_split = []
    test_split = []
    for path in paths:
        # If path is already in one of the splits, add it to the corresponding split
        # Otherwise add it to the splits with the corresponding percentage
        if path in past_train_files:
            train_split.append(path)
        elif path in past_val_files:
            val_split.append(path)
        elif path in past_test_files:
            test_split.append(path)
        else:
            x = randomizer.random()
            if x < TRAIN_SPLIT_PERCENTAGE:
                train_split.append(path)
            elif x < TRAIN_SPLIT_PERCENTAGE + VALIDATION_SPLIT_PERCENTAGE:
                val_split.append(path)
            else:
                test_split.append(path)

    if config.dataset_split_path:
        with open(config.dataset_split_path, "wb") as f:
            pickle.dump({"train": train_split, "val": val_split, "test": test_split}, f)
    return train_split, val_split, test_split


def read_random_audio_slice(file_path: str, slice_length: int, target_sr: int, naudios: int) -> list[Audio]:
    """
    Reads a random slice of an audio file with the specified length at the target sample rate.

    Args:
    file_path (str): Path to the audio file.
    slice_length (int): Length of the audio slice in samples.
    target_sr (int): Target sample rate for the audio slice.
    naudios (int): How many different slices to read

    Returns:
    audio: Audio object containing the audio data and sample rate with exactly the requested number of samples.
    """
    slices = []
    for i in range(naudios):
        with sf.SoundFile(file_path) as sf_file:
            total_samples = len(sf_file)
            if slice_length > total_samples:
                raise ValueError("Slice length is greater than total number of samples in the file.")
            start_sample = np.random.randint(0, total_samples - slice_length)
            sf_file.seek(start_sample)
            sample_rate = sf_file.samplerate
            # Intentionally read a little bit more so that resampled audio sounds hopefully less weird
            required_samples = int(slice_length * (sample_rate / target_sr)) + WAV_READ_BUFFER
            data = sf_file.read(frames=required_samples, dtype='float32')

        num_channels = sf_file.channels
        tensor_data = torch.tensor(data)
        if num_channels == 1:
            tensor_data = tensor_data.unsqueeze(0)
        else:
            tensor_data = tensor_data.transpose(0, 1)

        audio = Audio(tensor_data, sample_rate)
        audio = audio.resample(target_sr)
        audio = audio.slice_frames(0, slice_length)
        slices.append(audio)
    return slices


def make_model(config: TrainingConfig) -> Vocoder:
    """Creates a new model based on the PatchGAN architecture."""
    stfts = [
        make_stft(config.audio_length, feat) for feat in config.output_features
    ]
    model = Vocoder(
        config,
        stfts,
        list(config.output_channels),
        kernel_size=list(config.kernel_size),
        audio_length=config.audio_length,
    )
    return model


def validate(
    config: TrainingConfig,
    model: Vocoder,
    val_data_loader: DataLoader,
    reconstruction_loss: nn.Module,
    perceptual_loss: nn.Module,
    step_count: int,
    stft: STFT
):
    val_count_ = 0
    if step_count % config.val_steps != (1 if config.validate_at_step_1 else 0):
        return

    model.eval()
    with torch.no_grad():
        val_recon_losses = []
        val_spec_losses = []
        val_perceptual_losses = []
        for target_audio in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(config.val_count, len(val_data_loader))):
            val_count_ += 1
            if val_count_ > config.val_count:
                break

            target_audio = target_audio.flatten(0, 1)
            assert isinstance(target_audio, torch.Tensor)
            _assert(target_audio.dim() == 2)                                    # im shape: B, L
            _assert(target_audio.shape[1] == config.audio_length)               # im shape: B, L

            batch_size = target_audio.shape[0]
            target_audio = target_audio.float().to(device)
            target_spec = stft.mel(
                target_audio.float(), config.sample_rate, config.nmel
            )                                                                   # shape: B, nmel, T

            # Fetch autoencoders output(reconstructions)
            with autocast('cuda'):
                pred_audio: torch.Tensor = model(target_spec)

            _assert(pred_audio.shape == target_audio.shape)

            pred_spec = stft.mel(
                pred_audio.float(), config.sample_rate, config.nmel
            )

            #######################################

            with autocast('cuda'):
                recon_loss = reconstruction_loss(pred_audio, target_audio)
                spec_loss = reconstruction_loss(pred_spec, target_spec)

            val_recon_loss = recon_loss.item()
            val_recon_losses.append(val_recon_loss)

            val_spec_loss = spec_loss.item()
            val_spec_losses.append(val_spec_loss)

            val_lpips_loss = torch.mean(perceptual_loss(pred_audio, target_audio)).item()
            val_perceptual_losses.append(val_lpips_loss)

    wandb.log({
        "Val Reconstruction Loss": np.mean(val_recon_losses),
        "Val Spectrogram Loss": np.mean(val_spec_losses),
        "Val Perceptual Loss": np.mean(val_perceptual_losses),
    }, step=step_count)

    tqdm.write(f"Validation complete: Reconstruction loss: {np.mean(val_recon_losses)}, Perceptual Loss: {np.mean(val_perceptual_losses)}")
    model.train()


def train(config_path: str, start_from_iter: int = 0):
    config = TrainingConfig.load(config_path)
    train_paths, val_paths, _ = make_split(config)
    set_seed(config.seed)

    # Create the model and dataset #
    model: Vocoder = make_model(config).to(device)
    print(f"Starting from iteration {start_from_iter}")

    numel = 0
    for p in model.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))

    # Create the dataset
    train_dataset = COMP5421Dataset(config, train_paths)
    val_dataset = COMP5421Dataset(config, val_paths)
    print('Train dataset size: {}'.format(len(train_dataset)))

    print('Val dataset size: {}'.format(len(val_dataset)))

    print(f"Effective audio length: {config.audio_length / config.sample_rate} seconds")

    data_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=True
    )

    val_data_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers_dl,
        shuffle=False
    )

    os.makedirs(config.output_dir, exist_ok=True)

    reconstruction_loss = torch.nn.MSELoss()
    disc_loss = DiscriminatorLoss(config)
    perceptual_loss = load_lpips(config)

    discriminator = Discriminator(config).to(device)

    def warmup_lr_scheduler(optimizer, warmup_steps, base_lr):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    optimizer_d = Adam(discriminator.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))

    scheduler_d = warmup_lr_scheduler(optimizer_d, config.warmup_steps, config.autoencoder_lr)
    scheduler_g = warmup_lr_scheduler(optimizer_g, config.warmup_steps, config.autoencoder_lr)

    accelerator = Accelerator(mixed_precision="bf16")

    step_count = 0
    progress_bar = tqdm(total=config.steps + start_from_iter, desc="Training Progress")

    # Reload checkpoint
    if start_from_iter > 0:
        model_save_path = config.get_vocoder_save_path(start_from_iter)
        model_sd = torch.load(model_save_path)
        model.load_state_dict(model_sd)
        disc_save_path = config.get_disc_save_path(start_from_iter)
        disc_sd = torch.load(disc_save_path)
        discriminator.load_state_dict(disc_sd)
        step_count = start_from_iter
        progress_bar.update(start_from_iter)

    model, optimizer_g, data_loader, optimizer_d = accelerator.prepare(
        model, optimizer_g, data_loader, optimizer_d
    )

    stft = STFT(config.nfft, config.ntimeframes)

    wandb.init(
        # set the wandb project where this run will be logged
        project=config.run_name,
        config=config.asdict(),
    )

    model.train()

    while True:
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        stop_training: bool = False

        for target_audio in data_loader:
            step_count += 1
            progress_bar.update(1)
            if step_count >= config.steps + start_from_iter:
                stop_training = True
                break

            target_audio = target_audio.flatten(0, 1)

            assert isinstance(target_audio, torch.Tensor)
            _assert(target_audio.dim() == 2, f"Got wrong shape: {target_audio.shape}")  # im shape: B, L
            _assert(target_audio.shape[1] == config.audio_length, f"Got wrong shape: {target_audio.shape}")

            batch_size = target_audio.shape[0]
            target_audio = target_audio.float().to(device)
            target_spec = stft.mel(
                target_audio.float(), config.sample_rate, config.nmel
            )                                                                   # shape: B, nmel, T

            # Fetch autoencoders output(reconstructions)
            with autocast('cuda'):
                pred_audio: torch.Tensor = model(target_spec)

            _assert(pred_audio.shape == target_audio.shape, f"Got wrong shape: {pred_audio.shape} vs {target_audio.shape}")

            pred_spec = stft.mel(
                pred_audio.float(), config.sample_rate, config.nmel
            )

            losses = {}

            ######### Optimize Generator ##########
            # L2 Loss
            with autocast("cuda"):
                losses["Audio Reconstruction Loss"] = reconstruction_loss(pred_audio, target_audio)
                losses["Spectrogram Reconstruction Loss"] = reconstruction_loss(pred_spec, target_spec)

            # Adversarial loss only if disc_step_start steps passed
            if step_count > config.disc_start:
                with autocast('cuda'):
                    dgz = discriminator(pred_audio, pred_spec)
                    disc_fake_loss_aud, disc_fake_loss_spec = disc_loss(dgz)
                losses["Generator Audio Loss"] = disc_fake_loss_aud
                losses["Generator Spectrogram Loss"] = disc_fake_loss_spec
            else:
                losses["Generator Audio Loss"] = torch.tensor(0.).to(device)
                losses["Generator Spectrogram Loss"] = torch.tensor(0.).to(device)

            # Perceptual Loss
            losses["Perceptual Loss"] = torch.mean(perceptual_loss(pred_audio, target_audio))

            g_loss = (
                losses["Audio Reconstruction Loss"] +
                losses["Spectrogram Reconstruction Loss"] +
                losses["Perceptual Loss"] * config.perceptual_weight +
                losses["Generator Audio Loss"] * config.disc_g_loss_audio_weight +
                losses["Generator Spectrogram Loss"] * config.disc_g_loss_spec_weight
            ) / config.autoencoder_acc_steps

            accelerator.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            if step_count > config.disc_start:
                discriminator.train()
                with autocast('cuda'):
                    dgz = discriminator(pred_audio.detach(), pred_spec.detach())
                    dx = discriminator(target_audio, target_spec)
                disc_fake_loss_aud, disc_fake_loss_spec = disc_loss(dgz, dx)
                losses["Discriminator Audio Loss"] = disc_fake_loss_aud
                losses["Discriminator Spectrogram Loss"] = disc_fake_loss_spec
                disc_fake_loss = disc_fake_loss_aud + disc_fake_loss_spec
                disc_fake_loss /= config.autoencoder_acc_steps
                accelerator.backward(disc_fake_loss)
                if step_count % config.autoencoder_acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                discriminator.eval()
            else:
                losses["Discriminator Audio Loss"] = torch.tensor(0.).to(device)
                losses["Discriminator Spectrogram Loss"] = torch.tensor(0.).to(device)
            #####################################

            if step_count % config.autoencoder_acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()

            # Log losses
            wandb.log({k: v.item() for k, v in losses.items()} | {
                "Discriminator Learning Rate": optimizer_g.param_groups[0]['lr'],
                "Generator Learning Rate": optimizer_d.param_groups[0]['lr'],
            }, step=step_count)

            if step_count % config.save_steps == 0:
                model_save_path = TrainingConfig.get_vocoder_save_path(config, step_count)
                disc_save_path = TrainingConfig.get_disc_save_path(config, step_count)
                torch.save(model.state_dict(), model_save_path)
                torch.save(discriminator.state_dict(), disc_save_path)

            scheduler_g.step()
            scheduler_d.step()

            ########### Perform Validation #############
            validate(config, model, val_data_loader, reconstruction_loss, perceptual_loss, step_count, stft)

        # End of epoch. Clean up the gradients and losses and save the model
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        model_save_path = config.get_vocoder_save_path(step_count)
        disc_save_path = config.get_disc_save_path(step_count)
        torch.save(model.state_dict(), model_save_path)
        torch.save(discriminator.state_dict(), disc_save_path)

        if stop_training:
            break

    wandb.finish()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vocoder training')
    parser.add_argument('--config', dest='config_path', default='config.yaml', type=str)
    parser.add_argument('--start_iter', dest='start_iter', type=int, default=0)
    args = parser.parse_args()
    train(args.config_path, start_from_iter=args.start_iter)
