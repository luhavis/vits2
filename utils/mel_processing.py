import torch
import torchaudio.transforms as T
import torch.utils.data

spectrogram_basis = {}
mel_scale_basis = {}
mel_spectrogram_basis = {}

def spectral_norm(x: torch.Tensor, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val))

def wav_to_spect(y: torch.Tensor, n_fft, sample_rate, hop_length, win_length, center=False) -> torch.Tensor:
    assert torch.min(y) >= -1.0, f"min value is {torch.min(y)}"
    assert torch.max(y) <= 1.0, f"max value is {torch.max(y)}"

    global spectrogram_basis
    dtype_device = str(y.dtype) + "_" + str(y.device)
    hparams = str(y.dtype) + "_" + str(n_fft) + "_" + str(hop_length)

    if hparams not in spectrogram_basis:
        spectrogram_basis[hparams] = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=(n_fft - hop_length) // 2,
            power=1,
            center=center
        ).to(device=y.device, dtype=y.dtype)

    spec = spectrogram_basis[hparams](y)
    spec = torch.sqrt(spec.pow(2) + 1e-6)
    return spec

