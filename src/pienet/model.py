"""PyTorch implementation of the PiENet pitch estimation network.

A 1-D convolutional network with a gated/dilated residual trunk (WaveNet-style,
but non-causal) operating on framed raw waveform, and a softmax over
log-spaced F0 bins plus one "unvoiced" class.

This is a direct port of ``model_fundf.py`` from the TensorFlow 1 release. All
convolutions use symmetric padding, which is what TensorFlow's ``'SAME'``
padding does for odd kernels, so the two implementations agree numerically.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PiENetConfig

__all__ = ["PiENet", "GatedResidualBlock"]

PathLike = Union[str, "os.PathLike[str]"]

#: Name of the pre-trained model shipped with the package.
DEFAULT_CHECKPOINT = "gtaug"


class GatedResidualBlock(nn.Module):
    """One gated dilated convolution block with residual and skip outputs."""

    def __init__(self, channels: int, filter_width: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (filter_width - 1) // 2
        self.dilation = dilation
        self.filter_gate = nn.Conv1d(
            channels, 2 * channels, filter_width, dilation=dilation, padding=padding
        )
        self.skip = nn.Conv1d(channels, channels, 1)
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: ``x`` of shape ``(B, C, T)``. Returns ``(residual, skip)``."""
        y = self.filter_gate(x)
        filt, gate = y.chunk(2, dim=1)
        y = torch.tanh(filt) * torch.sigmoid(gate)
        skip = self.skip(y)
        return self.output(y) + x, skip


class PiENet(nn.Module):
    """Pitch estimation network.

    The forward pass maps framed waveform to per-frame class logits::

        (batch, n_frames, win_length) -> (batch, n_frames, n_bins)

    Example:
        >>> model = PiENet.from_pretrained()          # doctest: +SKIP
        >>> logits = model(frames)                    # doctest: +SKIP
    """

    def __init__(self, config: Optional[PiENetConfig] = None) -> None:
        super().__init__()
        self.config = config or PiENetConfig()
        c = self.config
        fw, r, s = c.filter_width, c.residual_channels, c.postnet_channels

        self.input_dropout = nn.Dropout(p=c.input_dropout)
        # 1x1 convolution over the frame: a learned linear transform of the
        # windowed waveform (input_channels == win_length).
        self.input_conv = nn.Conv1d(c.win_length, r, 1)
        self.blocks = nn.ModuleList(
            GatedResidualBlock(r, fw, d) for d in c.dilations
        )
        self.postnet_conv1 = nn.Conv1d(r, s, fw, padding=(fw - 1) // 2)
        self.postnet_conv2 = nn.Conv1d(s, c.n_bins, fw, padding=(fw - 1) // 2)

        self.reset_parameters()

    # ------------------------------------------------------------------ #

    def reset_parameters(self) -> None:
        """Glorot-uniform weights, zero biases (as in the original model)."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def receptive_field_frames(self) -> int:
        """One-sided receptive field in frames (see :class:`PiENetConfig`)."""
        return self.config.receptive_field_frames

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ #

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute per-frame F0 class logits.

        Args:
            frames: ``(batch, n_frames, win_length)`` framed waveform, or
                ``(n_frames, win_length)`` for a single unbatched utterance.

        Returns:
            Logits with the same leading dimensions and ``n_bins`` channels.
            Apply ``softmax`` over the last axis for probabilities.
        """
        squeeze = frames.dim() == 2
        if squeeze:
            frames = frames.unsqueeze(0)
        if frames.dim() != 3:
            raise ValueError(
                f"expected (batch, n_frames, win_length), got {tuple(frames.shape)}"
            )
        if frames.shape[-1] != self.config.win_length:
            raise ValueError(
                f"expected frames of length {self.config.win_length}, "
                f"got {frames.shape[-1]}"
            )

        x = frames.transpose(1, 2)  # -> (B, win_length, T)
        x = self.input_dropout(x)
        x = torch.tanh(self.input_conv(x))

        skips: List[torch.Tensor] = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)

        y = torch.stack(skips, dim=0).sum(dim=0)
        y = F.relu(self.postnet_conv1(y))
        y = self.postnet_conv2(y)

        y = y.transpose(1, 2)  # -> (B, T, n_bins)
        return y.squeeze(0) if squeeze else y

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: PathLike, **meta: Any) -> None:
        """Save weights and config to a single ``.pt`` file."""
        torch.save(
            {
                "format": "pienet-v1",
                "config": self.config.to_dict(),
                "state_dict": self.state_dict(),
                "meta": meta,
            },
            str(path),
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_name: PathLike = DEFAULT_CHECKPOINT,
        map_location: Any = "cpu",
        strict: bool = True,
    ) -> "PiENet":
        """Load a model from a ``.pt`` checkpoint or a bundled model name.

        Args:
            path_or_name: Path to a checkpoint, or the name of a checkpoint
                shipped in ``pienet/assets`` (default: ``"gtaug"``, the
                GTE-AUG model from the paper).
            map_location: Passed through to :func:`torch.load`.
            strict: Require an exact state-dict match.

        Returns:
            A model in ``eval()`` mode.
        """
        path = resolve_checkpoint(path_or_name)
        try:
            payload = torch.load(
                str(path), map_location=map_location, weights_only=True
            )
        except (TypeError, RuntimeError, AttributeError):
            # older torch without weights_only, or non-tensor metadata
            payload = torch.load(str(path), map_location=map_location)

        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError(f"{path} is not a PiENet checkpoint")

        config = PiENetConfig.from_dict(payload.get("config", {}))
        model = cls(config)
        model.load_state_dict(payload["state_dict"], strict=strict)
        model.eval()
        return model


def resolve_checkpoint(path_or_name: PathLike) -> str:
    """Resolve a checkpoint name or path to an existing file."""
    from pathlib import Path

    p = Path(str(path_or_name))
    if p.exists():
        return str(p)

    assets = Path(__file__).parent / "assets"
    for candidate in (assets / f"{p.name}.pt", assets / p.name):
        if candidate.exists():
            return str(candidate)

    available = sorted(f.stem for f in assets.glob("*.pt")) if assets.exists() else []
    raise FileNotFoundError(
        f"checkpoint {path_or_name!r} not found. "
        f"Bundled checkpoints: {available or 'none'}"
    )


def available_checkpoints() -> List[str]:
    """Names of the checkpoints bundled with the package."""
    from pathlib import Path

    assets = Path(__file__).parent / "assets"
    return sorted(f.stem for f in assets.glob("*.pt")) if assets.exists() else []
