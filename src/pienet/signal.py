"""Signal framing, F0 bin mapping and audio I/O helpers.

These are the PyTorch/NumPy replacements for the original ``sp_module.py``.
The framing and bin conventions are bit-for-bit compatible with the
TensorFlow 1 release, so pre-trained models keep working.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "frame_signal",
    "num_frames",
    "f0_bin_centers",
    "f0_to_bin_index",
    "f0_to_onehot",
    "activations_to_f0",
    "read_audio",
    "resample",
]

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #

def num_frames(n_samples: int, hop_length: int) -> int:
    """Number of analysis frames produced for a signal of ``n_samples``.

    Matches the original convention ``ceil(len(x) / hop)``.
    """
    if n_samples <= 0:
        return 0
    return int(np.ceil(n_samples / hop_length))


def frame_signal(x: np.ndarray, win_length: int, hop_length: int) -> np.ndarray:
    """Split a waveform into overlapping frames.

    The signal is zero-padded with ``win_length // 2`` samples at both ends and
    then sliced with the given hop, producing ``ceil(len(x) / hop)`` frames.
    This reproduces ``sp_module.get_frames`` from the TensorFlow release, but
    vectorised (no Python loop) and safe for short inputs.

    Args:
        x: 1-D waveform.
        win_length: Frame length in samples.
        hop_length: Frame shift in samples.

    Returns:
        Array of shape ``(n_frames, win_length)`` and dtype float32.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
    n = num_frames(x.shape[0], hop_length)
    if n == 0:
        return np.zeros((0, win_length), dtype=np.float32)

    pad = win_length // 2
    needed = (n - 1) * hop_length + win_length
    padded = np.zeros(max(needed, x.shape[0] + 2 * pad), dtype=np.float32)
    padded[pad : pad + x.shape[0]] = x

    frames = np.lib.stride_tricks.sliding_window_view(padded, win_length)[
        :: hop_length
    ][:n]
    return np.ascontiguousarray(frames, dtype=np.float32)


# --------------------------------------------------------------------------- #
# F0 <-> class bins
# --------------------------------------------------------------------------- #

def f0_bin_centers(
    f0_min: float = 50.0, f0_max: float = 500.0, n_bins: int = 351
) -> np.ndarray:
    """Log-spaced centre frequency of each voiced output bin.

    Returns ``n_bins - 1`` values; the remaining class encodes "unvoiced".
    """
    return np.exp(
        np.linspace(np.log(f0_min), np.log(f0_max), n_bins - 1)
    ).astype(np.float32)


def f0_to_bin_index(
    f0: np.ndarray,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    n_bins: int = 351,
) -> np.ndarray:
    """Map an F0 contour (Hz, 0 = unvoiced) to class indices.

    Values ``<= 0`` map to the unvoiced class ``n_bins - 1``. Voiced values map
    to the nearest log-spaced bin (nearest in linear Hz, as in the original).

    Returns:
        Int64 array of shape ``(n_frames,)``.
    """
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    centers = f0_bin_centers(f0_min, f0_max, n_bins)
    idx = np.abs(centers[None, :] - f0[:, None]).argmin(axis=1)
    idx = np.where(f0 > 0.0, idx, n_bins - 1)
    return idx.astype(np.int64)


def f0_to_onehot(
    f0: np.ndarray,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    n_bins: int = 351,
) -> np.ndarray:
    """One-hot encoding of an F0 contour, shape ``(n_frames, n_bins)``.

    Kept for compatibility with the original API; training uses
    :func:`f0_to_bin_index` with a cross-entropy loss instead, which is
    mathematically identical and cheaper.
    """
    idx = f0_to_bin_index(f0, f0_min, f0_max, n_bins)
    onehot = np.zeros((idx.shape[0], n_bins), dtype=np.float32)
    onehot[np.arange(idx.shape[0]), idx] = 1.0
    return onehot


def activations_to_f0(
    activations: np.ndarray,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
    n_bins: int = 351,
    voicing_threshold: Optional[float] = None,
    interpolate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode network outputs into an F0 contour.

    Args:
        activations: ``(n_frames, n_bins)`` logits or probabilities.
        f0_min, f0_max, n_bins: Bin layout, must match the model config.
        voicing_threshold: If ``None`` (default, and what the original code
            does) a frame is voiced when the arg-max class is a pitch bin.
            If a float in ``[0, 1]`` is given, a frame is voiced when
            ``1 - p(unvoiced) >= voicing_threshold``, which lets you trade
            voiced/unvoiced errors without retraining. Requires probabilities
            (pass softmax output, or use :class:`pienet.F0Estimator`).
        interpolate: If true, refine the arg-max bin by fitting a parabola to
            the neighbouring bins in the log-F0 domain. Off by default so the
            output matches the TensorFlow implementation exactly.

    Returns:
        ``(f0, voicing)`` where ``f0`` is float32 Hz with 0 for unvoiced
        frames, and ``voicing`` is ``1 - p(unvoiced)`` per frame (only
        meaningful when ``activations`` are probabilities).
    """
    act = np.asarray(activations, dtype=np.float32)
    if act.ndim != 2:
        raise ValueError(f"expected (n_frames, n_bins), got {act.shape}")
    if act.shape[1] != n_bins:
        raise ValueError(f"expected {n_bins} bins, got {act.shape[1]}")

    centers = f0_bin_centers(f0_min, f0_max, n_bins)
    unvoiced = n_bins - 1
    voicing = 1.0 - act[:, unvoiced]

    if voicing_threshold is None:
        idx = act.argmax(axis=1)
        voiced = idx < unvoiced
    else:
        idx = act[:, :unvoiced].argmax(axis=1)
        voiced = voicing >= voicing_threshold

    idx = np.clip(idx, 0, unvoiced - 1)
    f0 = centers[idx].astype(np.float32)

    if interpolate:
        log_centers = np.log(centers)
        step = float(log_centers[1] - log_centers[0])
        left = np.clip(idx - 1, 0, unvoiced - 1)
        right = np.clip(idx + 1, 0, unvoiced - 1)
        rows = np.arange(act.shape[0])
        a, b, c = act[rows, left], act[rows, idx], act[rows, right]
        denom = a - 2.0 * b + c
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
        shift = np.clip(shift, -0.5, 0.5)
        f0 = np.exp(log_centers[idx] + shift * step).astype(np.float32)

    f0 = np.where(voiced, f0, 0.0).astype(np.float32)
    return f0, voicing.astype(np.float32)


# --------------------------------------------------------------------------- #
# Audio I/O
# --------------------------------------------------------------------------- #

_INT_SCALE = {
    np.dtype("int16"): 2.0**15,
    np.dtype("int32"): 2.0**31,
}


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal using a polyphase filter."""
    if orig_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    from math import gcd

    import scipy.signal

    g = gcd(int(orig_sr), int(target_sr))
    up, down = int(target_sr) // g, int(orig_sr) // g
    return scipy.signal.resample_poly(x, up, down).astype(np.float32)


def read_audio(
    path: PathLike, target_sr: Optional[int] = 16000, mono: bool = True
) -> Tuple[np.ndarray, int]:
    """Read an audio file as float32 in ``[-1, 1]``.

    Uses :mod:`soundfile` when installed (any libsndfile format), otherwise
    falls back to :mod:`scipy.io.wavfile` (PCM/float WAV only). Integer PCM is
    normalised by its full scale, so 24/32-bit files no longer come out
    thousands of times too loud, which the original ``y / 2**15`` did.

    Args:
        path: Audio file.
        target_sr: Resample to this rate. ``None`` keeps the native rate.
        mono: Average multi-channel input down to one channel.

    Returns:
        ``(waveform, sample_rate)``.
    """
    try:
        import soundfile as sf  # type: ignore

        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        import scipy.io.wavfile as wavfile

        sr, y = wavfile.read(str(path))
        y = np.asarray(y)
        if y.dtype == np.uint8:  # 8-bit PCM is unsigned
            y = (y.astype(np.float32) - 128.0) / 128.0
        elif y.dtype in _INT_SCALE:
            y = y.astype(np.float32) / _INT_SCALE[y.dtype]
        else:
            y = y.astype(np.float32)

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1 and mono:
        y = y.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        y = resample(y, sr, target_sr)
        sr = target_sr

    return np.ascontiguousarray(y, dtype=np.float32), int(sr)
