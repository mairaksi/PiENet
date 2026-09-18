"""PiENet - noise-robust neural F0 (pitch) estimation for speech.

PyTorch port of the pitch estimation network from:

    M. Airaksinen, L. Juvela, P. Alku and O. Rasanen,
    "Data augmentation strategies for neural F0 estimation", ICASSP 2019.

Quick start::

    import pienet

    f0 = pienet.estimate_f0("speech.wav")          # Hz per 10 ms frame, 0 = unvoiced

    est = pienet.F0Estimator(device="cuda")        # reuse the model
    f0, voicing = est.estimate_file("speech.wav", return_voicing=True)
"""

from __future__ import annotations

from .config import PiENetConfig
from .inference import F0Estimator, estimate_f0
from .model import PiENet, available_checkpoints
from .signal import (
    activations_to_f0,
    f0_bin_centers,
    f0_to_bin_index,
    f0_to_onehot,
    frame_signal,
    read_audio,
)

__version__ = "2.0.0"

__all__ = [
    "PiENet",
    "PiENetConfig",
    "F0Estimator",
    "estimate_f0",
    "available_checkpoints",
    "frame_signal",
    "read_audio",
    "f0_bin_centers",
    "f0_to_bin_index",
    "f0_to_onehot",
    "activations_to_f0",
    "__version__",
]
