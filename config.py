import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torchaudio
import os
from dataclasses import dataclass, asdict
from typing import List
import yaml


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dirs: tuple[str, ...] = ("E:/audio-dataset-v3/audio", "./fma_small_mp3")
    output_dir: str = "E:/output"
    num_workers_dl: int = 2
    batch_size: int = 1
    ds_batch_size: int = 3

    sample_rate: int = 44100
    ntimeframes: int = 432
    nfft: int = 1025
    nmel: int = 128
    output_channels: tuple[int, ...] = (512, 128, 64, 8)
    kernel_size: tuple[int, ...] = (9, 7, 5, 3)
    stride: tuple[int, ...] = (4, 4, 4, 4)
    output_features: tuple[int, ...] = (512, 128, 64, 8)
    gradient_checkpointing: bool = True

    disc_start: int = 2048
    disc_spec_weight: float = 3
    nfilters: int = 1024
    naudio_disc_layers: int = 4
    audio_disc_downsampling_factor: int = 4
    nspec_disc_patches: int = 4

    seed: int = 1943
    val_count: int = 100
    disc_audio_weights: tuple[float, ...] = (1, 1, 1, 1)
    disc_g_loss_spec_weight: float = 0.25
    disc_g_loss_audio_weight: float = 0.25
    perceptual_weight: int = 1
    steps: int = 100000
    warmup_steps: int = 1000
    autoencoder_lr: float = 0.00001
    autoencoder_acc_steps: int = 16
    save_steps: int = 2048
    ckpt_name: str = 'vocoder.pth'
    run_name: str = "comp5421-training"
    disc_loss: str = "bce"
    val_steps: int = 512
    validate_at_step_1: bool = True

    @property
    def audio_length(self) -> int:
        return STFT(self.nfft, self.ntimeframes).l

    @property
    def num_audio_discriminators(self) -> int:
        return len(self.disc_audio_weights)

    @property
    def dataset_split_path(self) -> str:
        return os.path.join(self.dataset_dirs[0], "dataset_split.pkl")

    @staticmethod
    def load(file_path: str = "config.yaml") -> 'TrainingConfig':
        """Loads the TrainingConfig from a YAML file. By default loads the one inside resources/config"""
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        default_dict = TrainingConfig().asdict()
        extra_keys = set(yaml_data.keys()) - set(default_dict.keys())
        if extra_keys:
            raise ValueError(f"Unexpected keys in YAML configuration: {extra_keys}")

        extra_keys = set(default_dict.keys()) - set(yaml_data.keys())
        if extra_keys:
            raise ValueError(f"Missing keys in YAML configuration: {extra_keys}")

        merged_config = {**default_dict, **yaml_data}
        for k, v in merged_config.items():
            if isinstance(v, list) and isinstance(default_dict[k], tuple):
                merged_config[k] = tuple(v)
        return TrainingConfig(**merged_config)

    def __post_init__(self):
        assert self.disc_loss in ("bce", "mse", "hinge")
        assert self.nspec_disc_patches > 0

    def get_vocoder_save_path(self, step: int) -> str:
        path = os.path.join(self.output_dir, self.run_name, f"step-{step:06d}", self.ckpt_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_disc_save_path(self, step: int) -> str:
        path = os.path.join(self.output_dir, self.run_name, f"step-{step:06d}", "disc_ckpt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def asdict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


class STFT:
    """Try to enforce the constraint about the inputs and outputs of the stft such that input is always (B, L) and output is (B, N, T)"""

    def __init__(self, n: int, t: int):
        # Check that N must be one more than a power of 2 and n//4 > 1
        assert (n > 3) and (((n - 1) & (n - 2)) == 0), f"n must be one more than a power of 2, got {n}"
        self.n = n
        self.t = t
        self.n_fft = 2 * n - 2
        self.hop_length = n // 4
        self._skip_checks = False

    @property
    def l(self):
        # Check that L is solvable: T = 1 + L // hop_length
        return (self.t - 1) * self.hop_length

    def _assert_is_audio(self, x: torch.Tensor, bs=None):
        if self._skip_checks:
            return
        assert isinstance(x, torch.Tensor), f"Expected input to be a tensor, got {type(x)}"
        assert x.ndim == 2, f"Expected input to have 2 dimensions, got {x.ndim}"
        if bs is None:
            bs = x.shape[0]
        assert x.shape == (bs, self.l), f"Expected input to have shape (B, {self.l}), got {x.shape}"

    def _assert_is_spectrogram(self, z: torch.Tensor, bs=None):
        if self._skip_checks:
            return
        assert isinstance(z, torch.Tensor), f"Expected output to be a tensor, got {type(z)}"
        if bs is None:
            bs = z.shape[0]
        try:
            assert z.shape == (bs, self.n, self.t), f"Expected output to have shape (B, {self.n}, {self.t}), got {z.shape}"
            assert torch.is_complex(z), f"Expected output to be complex, got {z.dtype}"
        except AssertionError:
            assert z.shape == (bs, self.n, self.t, 2), f"Expected output to have shape (B, {self.n}, {self.t}), got {z.shape}"
            assert torch.is_floating_point(z), f"Expected output to be floating point, got {z.dtype}"

    def _assert_is_mel(self, mel: torch.Tensor, bs=None):
        if self._skip_checks:
            return
        assert isinstance(mel, torch.Tensor), f"Expected output to be a tensor, got {type(mel)}"
        if bs is None:
            bs = mel.shape[0]
        assert mel.shape[2] == self.t, f"Expected output to have shape (B, nmels, {self.t}), got {mel.shape}"
        assert torch.is_floating_point(mel), f"Expected output to be floating point, got {mel.dtype}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._assert_is_audio(x)

        z = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=True,
            onesided=True,
        )

        self._assert_is_spectrogram(z, bs=x.shape[0])
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        self._assert_is_spectrogram(z)

        if z.ndim == 4:
            z = torch.view_as_complex(z)

        # We expect no trimming or padding to be done, so length=None
        x = torch.istft(
            z,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(z.real),
            win_length=self.n_fft,
            normalized=True,
            center=True
        )

        self._assert_is_audio(x, bs=z.shape[0])
        return x

    def mel(self, x: torch.Tensor, sample_rate: int, nmels: int = 80) -> torch.Tensor:
        self._assert_is_audio(x)

        # Compute the mel filterbank
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            n_mels=nmels,
            center=True,
            pad_mode='reflect',
            power=2.0,
        )(x)

        self._assert_is_mel(mel, bs=x.shape[0])
        return mel

    def mel_librosa(self, x: torch.Tensor, sample_rate: int, nmels: int = 80):
        self._assert_is_audio(x)
        assert x.ndim == 2, f"Expected input to have 2 dimensions, got {x.ndim}"
        assert x.shape[1] == self.l, f"Expected input to have shape (B, {self.l}), got {x.shape}"

        stft_ = np.abs(librosa.stft(x.numpy(), n_fft=self.n_fft, hop_length=self.hop_length))
        mel = librosa.feature.melspectrogram(sr=sample_rate, S=stft_**2, n_mels=nmels)
        mel = torch.from_numpy(mel)

        self._assert_is_mel(mel, bs=x.shape[0])
        return mel

    def show_mel(self, mel: torch.Tensor, sample_rate: int):
        self._assert_is_mel(mel)
        fig, ax = plt.subplots()
        melspec = librosa.power_to_db(mel.detach().cpu().numpy()[0], ref=np.max)
        img = librosa.display.specshow(melspec, sr=sample_rate, x_axis='time', y_axis='mel', cmap='viridis')
        cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig, ax

    def show_spec(self, z: torch.Tensor, sample_rate: int):
        self._assert_is_spectrogram(z)
        fig, ax = plt.subplots()
        spec = librosa.amplitude_to_db(z.abs().detach().cpu().numpy()[0], ref=np.max)
        img = librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='linear', cmap='viridis')
        cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig, ax


def make_stft(
    audio_length: int | None = None,
    n: int | None = None,
    t: int | None = None,
    hop_length: int | None = None,
    ensure_relations: bool = True,
) -> STFT:
    """Make a STFT object with the given parameters. If ensure_relations is True, we will try to enforce the relations between the parameters."""
    if n is None and t is None:
        # Give a random default to n and move on
        n = 513

    # If audio_length is not None and the other ones are not None
    if n is not None and t is not None:
        stft = STFT(n, t)
        if hop_length is not None:
            stft.hop_length = hop_length
        if ensure_relations and audio_length is not None:
            assert stft.l == audio_length, f"Audio length must be {stft.l}, but got {audio_length}"
        stft._skip_checks = not ensure_relations
        return stft

    assert n is not None or t is not None, "Either n or t must be provided"
    assert audio_length is not None, "Audio length must be provided if exactly one of n or t are not provided"

    if n is not None:
        if hop_length is None:
            hop_length = n // 4
        t = 1 + audio_length // hop_length
        stft = STFT(n, t)
        stft.hop_length = hop_length
        assert stft.l == audio_length, f"Audio length must be {stft.l}, but got {audio_length}"
        stft._skip_checks = not ensure_relations
        return stft

    assert t is not None
    assert audio_length % (t - 1) == 0, "The audio length must be divisible by t - 1"
    assert t > 1, "t must be greater than 1"

    if hop_length is None:
        hop_length = audio_length // (t - 1)

    # Try to make more overlap by default - set n to the previous power of 2 of 4 * hop_length and then + 1
    n = 1 << np.floor(np.log2(4 * hop_length)) + 1
    assert n is not None

    stft = STFT(n, t)
    stft.hop_length = hop_length
    assert stft.l == audio_length, f"Audio length must be {stft.l}, but got {audio_length}"
    stft._skip_checks = not ensure_relations
    return stft
