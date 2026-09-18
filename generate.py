#!/usr/bin/env python3
"""Backwards-compatible wrapper for the original ``generate.py`` interface.

Kept so the command lines documented for the TensorFlow release keep working::

    python generate.py                               # -> test_wavs.scp, f0/
    python generate.py input.wav                     # -> f0/
    python generate.py input_list.scp target_dir/
    python generate.py input_dir/ target_dir/

New code should use the ``pienet`` command or the Python API::

    pienet estimate input_dir/ target_dir/
    import pienet; f0 = pienet.estimate_f0("input.wav")
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pienet.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 2:
        raise SystemExit("Too many input arguments")
    file_list = args[0] if len(args) >= 1 else "test_wavs.scp"
    target_dir = args[1] if len(args) >= 2 else "f0/"
    raise SystemExit(main(["estimate", file_list, target_dir]))
