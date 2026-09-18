"""Data augmentation for noise-robust F0 estimation.

Port of ``augmentation.py`` from the TensorFlow release. Two strategies from
the paper are implemented here: random convolutional (channel) noise and random
additive noise, either white or sampled from a noise corpus.

The vocoder-based ground-truth-enhancement and diversity augmentations from the
paper are not part of this release; they are applied offline (see the README).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from .signal import read_audio

__all__ = [
    "NoiseBank",
    "AugmentConfig",
    "load_noise_samples",
    "augment",
    "add_noise",
    "add_channel_noise",
]

PathLike = Union[str, "os.PathLike[str]"]


class NoiseBank:
    """In-memory bank of noise recordings to draw additive noise from."""

    def __init__(self, samples: Sequence[np.ndarray]) -> None:
        self.samples: List[np.ndarray] = [
            np.asarray(s, dtype=np.float32).reshape(-1) for s in samples
        ]
        if not self.samples:
            raise ValueError("noise bank is empty")

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def from_scp(
        cls, scp_path: PathLike, sample_rate: int = 16000
    ) -> "NoiseBank":
        """Load every file listed (one path per line) in an ``.scp`` file."""
        with open(scp_path) as fh:
            paths = [line.strip() for line in fh if line.strip()]
        if not paths:
            raise ValueError(f"{scp_path} lists no files")
        return cls([read_audio(p, target_sr=sample_rate)[0] for p in paths])

    def draw(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Random excerpt of ``n_samples``, looped if the source is shorter."""
        noise = self.samples[rng.integers(len(self.samples))]
        if noise.shape[0] < n_samples:
            reps = int(np.ceil(n_samples / max(noise.shape[0], 1)))
            noise = np.tile(noise, reps)
        start = int(rng.integers(0, noise.shape[0] - n_samples + 1))
        return noise[start : start + n_samples]


def load_noise_samples(
    noise_wav_scp: Optional[PathLike] = "noiselist.scp", sample_rate: int = 16000
) -> Optional[NoiseBank]:
    """Load a :class:`NoiseBank`, or ``None`` to fall back to white noise."""
    if noise_wav_scp is None:
        return None
    return NoiseBank.from_scp(noise_wav_scp, sample_rate=sample_rate)


@dataclass
class AugmentConfig:
    """Probabilities and ranges for the random augmentation pipeline.

    Defaults reproduce the ``GTE-AUG`` training recipe.

    Attributes:
        channel_noise_prob: Probability of applying a random FIR channel.
        channel_taps: Length of the random impulse response.
        additive_noise_prob: Probability of adding noise.
        snr_min_db, snr_max_db: Uniform SNR range, in dB (inclusive-exclusive
            integers, as in the original).
        gain_range_db: Random output gain drawn from
            ``10 ** (-(U(0,1) - 0.5))``, i.e. roughly +/- 10 dB. Applied only
            when additive noise was applied, matching the original code.
    """

    channel_noise_prob: float = 0.5
    channel_taps: int = 17
    additive_noise_prob: float = 0.5
    snr_min_db: int = -10
    snr_max_db: int = 20
    random_gain: bool = True


def add_channel_noise(
    x: np.ndarray, rng: np.random.Generator, taps: int = 17
) -> np.ndarray:
    """Convolve with a random impulse response with a unit main tap."""
    imp = rng.standard_normal(taps).astype(np.float32) * rng.random(dtype=np.float32)
    imp[taps // 2] = 1.0
    return np.convolve(x, imp, "same").astype(np.float32)


def add_noise(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    noise_bank: Optional[NoiseBank] = None,
) -> np.ndarray:
    """Add noise at a given SNR (white noise when no bank is given)."""
    x = np.asarray(x, dtype=np.float32)
    energy = np.linalg.norm(x)
    if energy == 0.0:
        return x
    noise = (
        rng.standard_normal(x.shape[0]).astype(np.float32)
        if noise_bank is None
        else noise_bank.draw(x.shape[0], rng)
    )
    noise_energy = np.linalg.norm(noise)
    if noise_energy == 0.0:
        return x
    gain = 10.0 ** (-snr_db / 20.0)
    return (x + gain * noise * energy / noise_energy).astype(np.float32)


def augment(
    x: np.ndarray,
    noise_bank: Optional[NoiseBank] = None,
    config: Optional[AugmentConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply the random augmentation chain to one utterance.

    Args:
        x: 1-D waveform.
        noise_bank: Source of additive noise; ``None`` uses white noise.
        config: Augmentation probabilities and ranges.
        rng: NumPy random generator, for reproducible augmentation.

    Returns:
        The augmented waveform (a new array; the input is not modified).
    """
    cfg = config or AugmentConfig()
    rng = rng or np.random.default_rng()
    y = np.array(x, dtype=np.float32, copy=True)

    if rng.random() < cfg.channel_noise_prob:
        y = add_channel_noise(y, rng, taps=cfg.channel_taps)

    if rng.random() < cfg.additive_noise_prob:
        snr = float(rng.integers(cfg.snr_min_db, cfg.snr_max_db))
        y = add_noise(y, snr, rng, noise_bank)
        if cfg.random_gain:
            y = (y * 10.0 ** (-(rng.random() - 0.5))).astype(np.float32)

    return y


def add_noise_controlled(
    x: np.ndarray,
    snr_db: float,
    noise_bank: Optional[NoiseBank] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Deterministic-SNR noise addition, for building test conditions."""
    return add_noise(x, snr_db, rng or np.random.default_rng(), noise_bank)
