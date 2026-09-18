#!/usr/bin/env python3
"""Backwards-compatible wrapper for the original ``train.py`` interface.

Running ``python train.py`` trains on ``train_wavs.scp`` / ``train_f0s.scp``
with the published recipe, as it did in the TensorFlow release.

New code should use the ``pienet`` command or the Python API::

    pienet train --wav-scp train_wavs.scp --f0-scp train_f0s.scp --amp
    from pienet.train import train_from_scp
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pienet.train import train_from_scp  # noqa: E402

if __name__ == "__main__":
    train_from_scp(
        wav_scp="train_wavs.scp",
        f0_scp="train_f0s.scp",
        # set to "noiselist.scp" to sample additive noise from a corpus
        noise_scp=None,
    )
