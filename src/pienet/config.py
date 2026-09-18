"""Model and signal configuration for PiENet."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple

__all__ = ["PiENetConfig"]


@dataclass(frozen=True)
class PiENetConfig:
    """Configuration of a PiENet pitch estimation network.

    The defaults reproduce the ``GTE-AUG`` model published with the original
    TensorFlow 1 release (Airaksinen et al., ICASSP 2019): 16 kHz speech,
    32 ms analysis window, 10 ms frame shift and a 50-500 Hz F0 range.

    Attributes:
        sample_rate: Sampling rate the model expects, in Hz.
        win_length: Analysis window length, in samples.
        hop_length: Frame shift, in samples.
        f0_min: Lowest F0 bin centre, in Hz.
        f0_max: Highest F0 bin centre, in Hz.
        n_bins: Number of output classes. ``n_bins - 1`` log-spaced pitch bins
            plus one trailing "unvoiced" class.
        residual_channels: Width of the residual/gated convolution trunk.
        postnet_channels: Width of the first post-processing convolution.
        filter_width: Kernel size of the dilated and post-processing
            convolutions. Must be odd so that padding stays symmetric.
        dilations: Dilation factor of each gated convolution block.
        input_dropout: Dropout applied to the framed waveform during training.
    """

    sample_rate: int = 16000
    win_length: int = 512
    hop_length: int = 160
    f0_min: float = 50.0
    f0_max: float = 500.0
    n_bins: int = 351
    residual_channels: int = 128
    postnet_channels: int = 256
    filter_width: int = 5
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8)
    input_dropout: float = 0.4

    def __post_init__(self) -> None:
        if self.filter_width % 2 != 1:
            raise ValueError(
                f"filter_width must be odd (TensorFlow 'SAME' padding is only "
                f"symmetric for odd kernels), got {self.filter_width}"
            )
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2 (one pitch bin + unvoiced)")
        if not 0.0 < self.f0_min < self.f0_max:
            raise ValueError("expected 0 < f0_min < f0_max")
        # dataclass is frozen; normalise dilations to a tuple via object.__setattr__
        object.__setattr__(self, "dilations", tuple(int(d) for d in self.dilations))

    # ------------------------------------------------------------------ #

    @property
    def n_pitch_bins(self) -> int:
        """Number of voiced pitch classes (the last class means 'unvoiced')."""
        return self.n_bins - 1

    @property
    def unvoiced_index(self) -> int:
        """Index of the unvoiced class in the output distribution."""
        return self.n_bins - 1

    @property
    def frame_rate(self) -> float:
        """Output frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def receptive_field_frames(self) -> int:
        """One-sided receptive field of the network, in frames.

        Every convolution uses symmetric padding, so an output frame depends on
        this many frames to its left and to its right. Used to chunk long
        signals without changing the result.
        """
        half = (self.filter_width - 1) // 2
        trunk = sum(half * d for d in self.dilations)
        postnet = 2 * half  # two post-processing convolutions, dilation 1
        return trunk + postnet

    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PiENetConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)
