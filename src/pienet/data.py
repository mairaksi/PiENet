"""Datasets for training PiENet on paired waveform / F0 files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import AugmentConfig, NoiseBank, augment
from .config import PiENetConfig
from .signal import f0_to_bin_index, frame_signal, num_frames, read_audio

__all__ = ["F0Dataset", "read_scp", "collate_batch", "Batch"]

PathLike = Union[str, "os.PathLike[str]"]

#: Ignored class index for padded frames in the cross-entropy loss.
PAD_INDEX = -100


def read_scp(path: PathLike) -> List[str]:
    """Read a Kaldi-style ``.scp`` list: one file path per line."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


@dataclass
class Batch:
    """A padded minibatch of framed audio and F0 targets."""

    frames: torch.Tensor  # (B, T, win_length) float32
    targets: torch.Tensor  # (B, T) int64, PAD_INDEX where padded
    lengths: torch.Tensor  # (B,) int64, true frame counts

    def to(self, device: Union[str, torch.device]) -> "Batch":
        return Batch(
            self.frames.to(device, non_blocking=True),
            self.targets.to(device, non_blocking=True),
            self.lengths.to(device, non_blocking=True),
        )


class F0Dataset(Dataset):
    """Paired waveform and reference-F0 dataset.

    Each item yields ``(frames, target_indices)`` for one utterance. Reference
    F0 files are raw ``float32`` binaries, one value per frame, with ``0``
    marking unvoiced frames - the format the original release used.

    The number of F0 values must match the framing convention
    ``n_frames = ceil(n_samples / hop_length)``; a mismatch of one or two
    frames is trimmed with a warning, anything larger raises.

    Args:
        wav_paths: Audio files.
        f0_paths: Matching reference F0 files.
        config: Model/signal configuration.
        augment_config: Augmentation settings; ``None`` disables augmentation
            (use this for the validation split).
        noise_bank: Additive noise source; ``None`` uses white noise.
        f0_downsample: Take every n-th F0 value. Use ``2`` when the reference
            was computed with a 5 ms hop and the model uses 10 ms.
        max_frames: Randomly crop longer utterances to this many frames, which
            keeps batches a manageable size.
        seed: Base seed for reproducible augmentation.
    """

    def __init__(
        self,
        wav_paths: Sequence[PathLike],
        f0_paths: Sequence[PathLike],
        config: Optional[PiENetConfig] = None,
        augment_config: Optional[AugmentConfig] = None,
        noise_bank: Optional[NoiseBank] = None,
        f0_downsample: int = 1,
        max_frames: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if len(wav_paths) != len(f0_paths):
            raise ValueError(
                f"got {len(wav_paths)} wav files but {len(f0_paths)} f0 files"
            )
        if not wav_paths:
            raise ValueError("dataset is empty")
        self.wav_paths = [str(p) for p in wav_paths]
        self.f0_paths = [str(p) for p in f0_paths]
        self.config = config or PiENetConfig()
        self.augment_config = augment_config
        self.noise_bank = noise_bank
        self.f0_downsample = int(f0_downsample)
        self.max_frames = max_frames
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.wav_paths)

    def set_epoch(self, epoch: int) -> None:
        """Vary the augmentation stream across epochs, reproducibly."""
        self.epoch = int(epoch)

    # ------------------------------------------------------------------ #

    def _load_f0(self, path: str, expected: int) -> np.ndarray:
        f0 = np.fromfile(path, dtype=np.float32)
        if self.f0_downsample > 1:
            f0 = f0[:: self.f0_downsample]
        if f0.shape[0] != expected:
            if abs(f0.shape[0] - expected) > 2:
                raise ValueError(
                    f"{path}: {f0.shape[0]} F0 values but {expected} frames "
                    f"expected. Check the hop size and the framing convention "
                    f"(n_frames = ceil(n_samples / hop))."
                )
            if f0.shape[0] > expected:
                f0 = f0[:expected]
            else:
                f0 = np.pad(f0, (0, expected - f0.shape[0]))
        return f0

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        y, _ = read_audio(self.wav_paths[index], target_sr=c.sample_rate)
        n = num_frames(y.shape[0], c.hop_length)
        f0 = self._load_f0(self.f0_paths[index], n)

        if self.augment_config is not None:
            rng = np.random.default_rng(
                (self.seed, self.epoch, index)
            )
            y = augment(y, self.noise_bank, self.augment_config, rng)

        frames = frame_signal(y, c.win_length, c.hop_length)
        targets = f0_to_bin_index(f0, c.f0_min, c.f0_max, c.n_bins)

        if self.max_frames is not None and frames.shape[0] > self.max_frames:
            rng = np.random.default_rng((self.seed, self.epoch, index, 1))
            start = int(rng.integers(0, frames.shape[0] - self.max_frames + 1))
            frames = frames[start : start + self.max_frames]
            targets = targets[start : start + self.max_frames]

        return torch.from_numpy(frames), torch.from_numpy(targets)


def collate_batch(items: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    """Pad a list of ``(frames, targets)`` pairs into a :class:`Batch`."""
    lengths = torch.tensor([f.shape[0] for f, _ in items], dtype=torch.long)
    max_len = int(lengths.max())
    win = items[0][0].shape[1]

    frames = torch.zeros(len(items), max_len, win, dtype=torch.float32)
    targets = torch.full((len(items), max_len), PAD_INDEX, dtype=torch.long)
    for i, (f, t) in enumerate(items):
        frames[i, : f.shape[0]] = f
        targets[i, : t.shape[0]] = t
    return Batch(frames=frames, targets=targets, lengths=lengths)
