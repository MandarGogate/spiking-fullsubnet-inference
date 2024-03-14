from typing import Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor

EPSILON = np.finfo(float).eps


def mag_phase(complex_valued_tensor: Tensor) -> tuple[Tensor, Tensor]:
    """Get magnitude and phase of a complex-valued tensor.

    Args:
        complex_valued_tensor: complex-valued tensor.

    Returns:
        magnitude and phase spectrogram.
    """
    mag, phase = torch.abs(complex_valued_tensor), torch.angle(complex_valued_tensor)
    return mag, phase


def stft(
        y: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        output_type=None,
) -> Union[Tensor, tuple[Tensor, ...]]:
    """Wrapper of the official ``torch.stft`` for single-channel and multichannel signals.

    Args:
        y: single-/multichannel signals with shape of [B, C, T] or [B, T].
        n_fft: num of FFT.
        hop_length: hop length.
        win_length: hanning window size.
        output_type: "mag_phase", "real_imag", "complex", or None.

    Returns:
        If the input is single-channel, return the spectrogram with shape of [B, F, T], otherwise [B, C, F, T].
        If output_type is "mag_phase", return a list of magnitude and phase spectrogram.
        If output_type is "real_imag", return a list of real and imag spectrogram.
        If output_type is None, return a list of magnitude, phase, real, and imag spectrogram.
    """
    ndim = y.dim()
    assert ndim in [2, 3], f"Only support single-/multi-channel signals. {ndim=}."

    batch_size, *_, num_samples = y.shape

    # Compatible with multi-channel signals
    if ndim == 3:
        y = y.reshape(-1, num_samples)

    complex_valued_stft = torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=y.device),
        return_complex=True,
    )

    _, num_freqs, num_frames = complex_valued_stft.shape

    if ndim == 3:
        complex_valued_stft = complex_valued_stft.reshape(
            batch_size, -1, num_freqs, num_frames
        )

    if output_type:
        if output_type == "mag_phase":
            return mag_phase(complex_valued_stft)
        elif output_type == "real_imag":
            return complex_valued_stft.real, complex_valued_stft.imag
        elif output_type == "complex":
            return complex_valued_stft
        else:
            raise NotImplementedError(
                "Only 'mag_phase', 'real_imag', and 'complex' are supported"
            )
    else:
        return (
            *mag_phase(complex_valued_stft),
            complex_valued_stft.real,
            complex_valued_stft.imag,
        )


def istft(
        feature: Union[Tensor, tuple[Tensor, ...], list[Tensor]],
        n_fft: int,
        hop_length: int,
        win_length: int,
        length: Optional[int] = None,
        input_type: Literal["mag_phase", "real_imag", "complex"] = "mag_phase",
) -> Tensor:
    """Wrapper of the official ``torch.istft`` for single-channel signals.

    Args:
        features: single-channel spectrogram with shape of [B, F, T] for input_type="complex" or ([B, F, T], [B, F, T]) for input_type="real_imag" and "mag_phase".
        n_fft: num of FFT.
        hop_length: hop length.
        win_length: hanning window size.
        length: expected length of istft.
        input_type: "real_image", "complex", or "mag_phase".

    Returns:
        Single-channel singal with the shape shape of [B, T].

    Notes:
        Only support single-channel input with shape of [B, F, T] or ([B, F, T], [B, F, T])
    """
    if input_type == "real_imag":
        assert isinstance(feature, tuple) or isinstance(
            feature, list
        )  # (real, imag) or [real, imag]
        real, imag = feature
        complex_valued_features = torch.complex(real=real, imag=imag)
    elif input_type == "complex":
        assert isinstance(feature, Tensor) and torch.is_complex(feature)
        complex_valued_features = feature
    elif input_type == "mag_phase":
        assert isinstance(feature, tuple) or isinstance(
            feature, list
        )  # (mag, phase) or [mag, phase]
        mag, phase = feature
        complex_valued_features = torch.complex(
            mag * torch.cos(phase), mag * torch.sin(phase)
        )
    else:
        raise NotImplementedError(
            "Only 'real_imag', 'complex', and 'mag_phase' are supported now."
        )

    return torch.istft(
        complex_valued_features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft, device=complex_valued_features.device),
        length=length,
    )
