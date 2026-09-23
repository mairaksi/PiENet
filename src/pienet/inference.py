"""F0 estimation with a trained PiENet model."""

from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .config import PiENetConfig
from .model import DEFAULT_CHECKPOINT, PiENet
from .signal import activations_to_f0, frame_signal, read_audio, resample

__all__ = ["F0Estimator", "estimate_f0"]

PathLike = Union[str, "os.PathLike[str]"]


def _pick_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class F0Estimator:
    """Reusable F0 estimator wrapping a :class:`~pienet.model.PiENet`.

    Loading the model is the expensive part, so keep one estimator around when
    processing many files.

    Args:
        model: A loaded model, a path to a ``.pt`` checkpoint, or the name of a
            bundled checkpoint (default ``"gtaug"``).
        device: Torch device. ``None`` picks CUDA, then MPS, then CPU.
        dtype: Compute dtype. ``torch.float16``/``bfloat16`` speed up GPU
            inference; decoding always happens in float32.
        chunk_frames: Process at most this many frames at a time, with enough
            overlap to keep the result identical to processing the whole
            signal at once. Keeps memory bounded on very long recordings.

    Example:
        >>> est = F0Estimator()                        # doctest: +SKIP
        >>> f0 = est.estimate_file("speech.wav")       # doctest: +SKIP
    """

    def __init__(
        self,
        model: Union[PiENet, PathLike] = DEFAULT_CHECKPOINT,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        chunk_frames: int = 20000,
    ) -> None:
        self.device = _pick_device(device)
        if isinstance(model, PiENet):
            self.model = model
        else:
            self.model = PiENet.from_pretrained(model, map_location="cpu")
        self.model = self.model.to(self.device).eval()
        self.dtype = dtype
        if dtype is not None:
            self.model = self.model.to(dtype)
        self.chunk_frames = int(chunk_frames)
        if self.chunk_frames <= 2 * self.model.receptive_field_frames:
            raise ValueError(
                f"chunk_frames must exceed twice the receptive field "
                f"({2 * self.model.receptive_field_frames})"
            )

    # ------------------------------------------------------------------ #

    @property
    def config(self) -> PiENetConfig:
        return self.model.config

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def probabilities(self, frames: np.ndarray) -> np.ndarray:
        """Per-frame class probabilities for one utterance.

        Args:
            frames: ``(n_frames, win_length)`` framed waveform.

        Returns:
            Float32 array ``(n_frames, n_bins)``.
        """
        frames_t = torch.as_tensor(frames, dtype=torch.float32)
        if frames_t.dim() != 2:
            raise ValueError(f"expected (n_frames, win_length), got {frames.shape}")
        n = frames_t.shape[0]
        if n == 0:
            return np.zeros((0, self.config.n_bins), dtype=np.float32)

        rf = self.model.receptive_field_frames
        out = torch.empty(n, self.config.n_bins, dtype=torch.float32)

        step = self.chunk_frames - 2 * rf
        for start in range(0, n, step):
            lo = max(0, start - rf)
            hi = min(n, start + step + rf)
            block = frames_t[lo:hi].unsqueeze(0).to(self.device)
            if self.dtype is not None:
                block = block.to(self.dtype)
            logits = self.model(block).float().squeeze(0).cpu()
            probs = torch.softmax(logits, dim=-1)
            keep_lo = start - lo
            keep_hi = keep_lo + min(step, n - start)
            out[start : start + (keep_hi - keep_lo)] = probs[keep_lo:keep_hi]

        return out.numpy()

    # ------------------------------------------------------------------ #

    def estimate(
        self,
        waveform: np.ndarray,
        sample_rate: Optional[int] = None,
        return_voicing: bool = False,
        voicing_threshold: Optional[float] = None,
        interpolate: bool = False,
        postprocess: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Estimate F0 from a waveform.

        Args:
            waveform: 1-D float array in ``[-1, 1]`` (integer input is scaled).
            sample_rate: Rate of ``waveform``; resampled to the model rate when
                it differs. ``None`` assumes the model rate.
            return_voicing: Also return ``1 - p(unvoiced)`` per frame.
            voicing_threshold: See :func:`pienet.signal.activations_to_f0`.
            interpolate: Parabolic sub-bin refinement of the F0 estimate.

        Returns:
            ``f0`` in Hz (0 = unvoiced), one value per ``hop_length`` samples,
            or ``(f0, voicing)`` when ``return_voicing`` is set.
        """
        c = self.config
        y = np.asarray(waveform)
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.float32) / float(np.iinfo(y.dtype).max + 1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if sample_rate is not None and sample_rate != c.sample_rate:
            y = resample(y, sample_rate, c.sample_rate)

        frames = frame_signal(y, c.win_length, c.hop_length)
        probs = self.probabilities(frames)
        f0, voicing = activations_to_f0(
            probs,
            f0_min=c.f0_min,
            f0_max=c.f0_max,
            n_bins=c.n_bins,
            voicing_threshold=voicing_threshold,
            interpolate=interpolate,
            postprocess=postprocess,
        )
        return (f0, voicing) if return_voicing else f0

    def estimate_file(
        self, path: PathLike, **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Estimate F0 from an audio file (resampled to the model rate)."""
        y, sr = read_audio(path, target_sr=self.config.sample_rate)
        return self.estimate(y, sample_rate=sr, **kwargs)

    def estimate_files(
        self, paths: Iterable[PathLike], **kwargs: Any
    ) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Estimate F0 for several files, reusing the loaded model."""
        return [self.estimate_file(p, **kwargs) for p in paths]


def estimate_f0(
    source: Union[PathLike, np.ndarray],
    sample_rate: Optional[int] = None,
    model: Union[PiENet, PathLike] = DEFAULT_CHECKPOINT,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """One-shot F0 estimation for a file path or waveform.

    Convenience wrapper that builds an :class:`F0Estimator` per call; use the
    class directly when processing more than a couple of files.

    Example:
        >>> f0 = estimate_f0("speech.wav")             # doctest: +SKIP
    """
    est = F0Estimator(model=model, device=device)
    if isinstance(source, (str, os.PathLike)):
        return est.estimate_file(source, **kwargs)
    return est.estimate(source, sample_rate=sample_rate, **kwargs)
