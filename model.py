import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np
from .config import TrainingConfig
from dataclasses import dataclass
from torchaudio.prototype.pipelines import VGGISH
from AutoMasher.fyp import Audio
import torchaudio
from .config import STFT, make_stft


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LearnableAdaptivePooling2d(nn.Module):
    def __init__(self, channels: int, input_size_x: int, input_size_y: int, output_size_x: int, output_size_y: int):
        super(LearnableAdaptivePooling2d, self).__init__()
        self.input_size_x = input_size_x
        self.input_size_y = input_size_y
        self.output_size_x = output_size_x
        self.output_size_y = output_size_y

        # Initialize the weights for the pooling layer
        stride_x = self.input_size_x // self.output_size_x
        kernel_size_x = self.input_size_x - (self.output_size_x - 1) * stride_x
        stride_y = self.input_size_y // self.output_size_y
        kernel_size_y = self.input_size_y - (self.output_size_y - 1) * stride_y

        self.pooling = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size_x, kernel_size_y),
            stride=(stride_x, stride_y),
            padding=0,
            bias=False
        )

        nn.init.constant_(self.pooling.weight, 1.0 / (kernel_size_x * kernel_size_y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == self.input_size_x and x.shape[3] == self.input_size_y, f"Input shape mismatch: {x.shape} vs {(x.shape[0], x.shape[1], self.input_size_x, self.input_size_y)}"
        x = self.pooling(x)
        assert x.shape[2] == self.output_size_x and x.shape[3] == self.output_size_y, f"Output shape mismatch: {x.shape} vs {(x.shape[0], x.shape[1], self.output_size_x, self.output_size_y)}"
        return x


class LearnableAdaptiveResnetBlock(nn.Module):
    """A ResNet block with learnable adaptive pooling.
    - Upscale block to the desired outout size
    - Convolutional layers with weight normalization
    - Residual connection
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, output_size_x: int, output_size_y: int):
        super(LearnableAdaptiveResnetBlock, self).__init__()
        self.pool = LearnableAdaptivePooling2d(in_channels, 1, 1, output_size_x, output_size_y)
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == self.pool.input_size_x and x.shape[3] == self.pool.input_size_y, f"Input shape mismatch: {x.shape} vs {(x.shape[0], x.shape[1], self.pool.input_size_x, self.pool.input_size_y)}"
        x = self.pool(x)
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return F.relu(x)


class Vocoder(nn.Module):
    """A simple vocoder model with learnable adaptive pooling and residual connections."""

    def __init__(
        self,
        stfts: list[STFT],
        output_channels: list[int],
        kernel_size: list[int],
        stride: list[int],
        audio_length: int,
    ):
        super(Vocoder, self).__init__()
        self.stfts = stfts
        self.audio_length = audio_length
        lengths = [stft.l for stft in self.stfts]
        assert all(length == audio_length for length in lengths), f"All STFTs must have the same length, but got {lengths}."
        assert len(stfts) == len(output_channels) == len(kernel_size) - 1 == len(stride) - 1, "All input lists must have the same length."
        resnet_blocks = []
        output_convs = []
        in_channels = 1
        for i in range(len(output_channels)):
            out_channels = output_channels[i]
            out_size_x = self.stfts[i].n
            out_size_y = self.stfts[i].t
            resnet_blocks.append(LearnableAdaptiveResnetBlock(in_channels, out_channels, kernel_size[i], stride[i], out_size_x, out_size_y))
            in_channels = out_channels
            output_convs.append(nn.Conv2d(out_channels, 2, kernel_size=1))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.output_convs = nn.ModuleList(output_convs)
        self.final_block = LearnableAdaptiveResnetBlock(in_channels, 1, kernel_size[-1], stride[-1], audio_length, 1)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, nfft, ntimeframes)
        assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
        x = x.unsqueeze(1)                    # Add channel dimension: (B, 1, nfft, ntimeframes)
        y: torch.Tensor = None                # type: ignore[assignment]
        for i, block in enumerate(self.resnet_blocks):
            x = block(x)                      # (B, out_channels, nfft, ntimeframes)
            out = self.output_convs[i](x)     # (B, 2, nfft, ntimeframes)
            out = out.permute(0, 2, 3, 1)     # (B, nfft, ntimeframes, 2)
            out = out.float()                 # (B, nfft, ntimeframes, 2)
            out = self.stfts[i].inverse(out)  # (B, L)
            if y is None:
                y = out
            else:
                y += out
        x = self.final_block(x)               # (B, 1, L, 1)
        x = x.squeeze(1).squeeze(-1)          # (B, L)
        return F.tanh(x + y)                  # (B, L), fix between [-1, 1] - TODO see if this is necessary


class Vggish(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_sr = VGGISH.sample_rate
        self.input_proc = VGGISH.get_input_processor()
        self.model = VGGISH.get_model()

    def forward(self, audio: Audio | tuple[torch.Tensor, int]):
        if isinstance(audio, Audio):
            x = audio.resample(self.input_sr).data
        else:
            y, sr = audio
            if sr != self.input_sr:
                x = torchaudio.functional.resample(y, sr, self.input_sr)
            else:
                x = y
        assert len(x.shape) in (2, 3), f"Expected 2D or 3D tensor, got {x.shape}"
        has_batch = len(x.shape) == 3
        if has_batch:
            b, c, t = x.shape
            x = x.view(-1, t)
        else:
            # To apease the typechecker
            b = 1
            c, t = x.shape
        feats = torch.stack([self.model(self.input_proc(w)) for w in x]).flatten(-2, -1)
        if has_batch:
            feats = feats.view(b, c, -1)
        return feats

    def __call__(self, audio: Audio | tuple[torch.Tensor, int]) -> torch.Tensor:
        return super().__call__(audio)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf: int, n_layers: int, downsampling_factor: int):
        super().__init__()
        model = []

        model.append(nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        ))

        nf = ndf
        nf_prev = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model.append(nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            ))

        nf = min(nf * 2, 1024)
        model.append(nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        ))

        model.append(WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        ))

        self.model = nn.ModuleList(model)

    def forward(self, x):
        # Expect input shape: (B, source, L)
        for layer in self.model:
            x = layer(x)
        return x.squeeze(1)


class AudioDiscriminator(nn.Module):
    def __init__(self, ndiscriminators: int, nfilters: int, n_layers: int, downsampling_factor: int):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(ndiscriminators):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                nfilters, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, audio: torch.Tensor):
        results = []
        for key, disc in self.model.items():
            results.append(disc(audio))
            audio = self.downsample(audio)
        return results


class SpectrogramPatchModel(nn.Module):
    """This uses the idea of PatchGAN but changes the architecture to use Conv2d layers on each audio snippet"""

    def __init__(self, nsources: int, nfeatures: int, ntimeframes: int, num_patches: int = 4):
        super(SpectrogramPatchModel, self).__init__()
        assert ntimeframes % num_patches == 0, "ntimeframes must be divisible by num_patches"

        # Define a simple CNN architecture for each patch
        self.conv1 = nn.Conv2d(nsources, 16, kernel_size=3, padding=1)
        self.pool11 = nn.AdaptiveMaxPool2d((nfeatures // 2, ntimeframes // num_patches))
        self.pool12 = nn.AdaptiveAvgPool2d((nfeatures // 2, ntimeframes // (num_patches * 2)))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool21 = nn.AdaptiveMaxPool2d((nfeatures // 4, ntimeframes // (num_patches * 2)))
        self.pool22 = nn.AdaptiveAvgPool2d((nfeatures // 4, ntimeframes // (num_patches * 4)))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool31 = nn.AdaptiveMaxPool2d((nfeatures // 8, ntimeframes // (num_patches * 4)))
        self.pool32 = nn.AdaptiveAvgPool2d((nfeatures // 8, ntimeframes // (num_patches * 8)))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Conv2d(128, 1, (nfeatures // 8, ntimeframes // (num_patches * 8)))  # Equivalent to FC layers over each channel
        self.nsources = nsources
        self.nfeatures = nfeatures
        self.ntimeframes = ntimeframes
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor):
        # x shape: (B, nsources, nfft, ntimeframes)
        batch_size = x.size(0)
        assert x.shape == (batch_size, self.nsources, self.nfeatures, self.ntimeframes), f"Input shape mismatch: {x.shape} vs {(batch_size, self.nsources, self.nfeatures, self.ntimeframes)}"
        # Splitting along the T axis into 4 patches
        x = x.unflatten(3, (self.num_patches, self.ntimeframes // self.num_patches)).permute(0, 3, 1, 2, 4)  # (B, npatch, nsources, nfft, ntimeframes // npatch)
        x = x.flatten(0, 1).contiguous()

        # Apply CNN
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool11(x)
        x = self.pool12(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool21(x)
        x = self.pool22(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool31(x)
        x = self.pool32(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(batch_size, self.num_patches, -1).squeeze(-1)
        return x


@dataclass
class MultiResolutionDiscriminatorOutput:
    audio_results: list[torch.Tensor]
    spectrogram_results: torch.Tensor


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.audio_discriminator = AudioDiscriminator(
            config.num_audio_discriminators,
            config.nfilters,
            config.naudio_disc_layers,
            config.audio_disc_downsampling_factor
        )
        self.spectrogram_discriminator = SpectrogramPatchModel(
            1, config.nmel, config.ntimeframes, config.nspec_disc_patches
        )

    def forward(self, audio: torch.Tensor, spectrogram: torch.Tensor) -> MultiResolutionDiscriminatorOutput:
        # Pass audio through the audio discriminator
        audio_results = self.audio_discriminator(audio)

        # Pass spectrogram through the spectrogram discriminator
        spectrogram_results = self.spectrogram_discriminator(spectrogram.unsqueeze(1))  # Unsqueeze because we assume one source which is not expressed in the original code

        return MultiResolutionDiscriminatorOutput(
            audio_results=audio_results,
            spectrogram_results=spectrogram_results,
        )

    def __call__(self, audio: torch.Tensor, spectrogram: torch.Tensor) -> MultiResolutionDiscriminatorOutput:
        # Exists purely for type annotation
        return super().__call__(audio, spectrogram)
