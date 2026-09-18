#!/usr/bin/env python3
"""Convert a TensorFlow 1 PiENet checkpoint into a PyTorch ``.pt`` file.

The original models were saved as TF1 ``tf.train.Saver`` checkpoints
(``saved_models/<name>_best.ckpt``). This script reads the variables with
TensorFlow 2's checkpoint reader (no TF1 graph, no ``tf.contrib``) and maps
them onto the PyTorch module.

Weight layout: TensorFlow stores 1-D convolution kernels as
``(filter_width, in_channels, out_channels)`` and PyTorch as
``(out_channels, in_channels, filter_width)``, so kernels are transposed
``(2, 1, 0)``. Both frameworks compute cross-correlation, so no kernel flip
is needed.

Usage::

    pip install "tensorflow-cpu>=2.12"
    python scripts/convert_tf_checkpoint.py \
        --tf-checkpoint legacy/saved_models/gtaug_best.ckpt \
        --output src/pienet/assets/gtaug.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pienet.config import PiENetConfig  # noqa: E402
from pienet.model import PiENet  # noqa: E402


def _kernel(w: np.ndarray) -> torch.Tensor:
    """(filter_width, in, out) -> (out, in, filter_width)."""
    return torch.from_numpy(np.ascontiguousarray(np.transpose(w, (2, 1, 0))))


def _bias(b: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(b.reshape(-1)))


def build_state_dict(
    tf_vars: Dict[str, np.ndarray], config: PiENetConfig, scope: str = "f0model"
) -> Dict[str, torch.Tensor]:
    """Map TensorFlow variable arrays onto PiENet's PyTorch state dict."""

    def var(name: str) -> np.ndarray:
        key = f"{scope}/{name}"
        if key not in tf_vars:
            raise KeyError(
                f"variable {key!r} missing from checkpoint; "
                f"found {len(tf_vars)} variables under other names"
            )
        return tf_vars[key]

    state: Dict[str, torch.Tensor] = {
        "input_conv.weight": _kernel(var("input_layer/W")),
        "input_conv.bias": _bias(var("input_layer/b")),
        "postnet_conv1.weight": _kernel(var("postproc_module/W1")),
        "postnet_conv1.bias": _bias(var("postproc_module/b1")),
        "postnet_conv2.weight": _kernel(var("postproc_module/W2")),
        "postnet_conv2.bias": _bias(var("postproc_module/b2")),
    }

    for i in range(len(config.dilations)):
        p = f"conv_modules/module{i}"
        state[f"blocks.{i}.filter_gate.weight"] = _kernel(var(f"{p}/filter_gate_W"))
        state[f"blocks.{i}.filter_gate.bias"] = _bias(var(f"{p}/filter_gate_b"))
        state[f"blocks.{i}.skip.weight"] = _kernel(var(f"{p}/skip_weight_W"))
        state[f"blocks.{i}.skip.bias"] = _bias(var(f"{p}/skip_weight_b"))
        state[f"blocks.{i}.output.weight"] = _kernel(var(f"{p}/output_weight_W"))
        state[f"blocks.{i}.output.bias"] = _bias(var(f"{p}/output_weight_b"))

    return state


def read_tf_checkpoint(path: str) -> Dict[str, np.ndarray]:
    """Read all non-optimizer variables from a TF1/TF2 checkpoint."""
    import tensorflow as tf  # imported lazily: only needed for conversion

    reader = tf.train.load_checkpoint(path)
    out: Dict[str, np.ndarray] = {}
    for name in reader.get_variable_to_shape_map():
        # skip Adam slot variables and global step counters
        if "/Adam" in name or name in ("beta1_power", "beta2_power", "global_step"):
            continue
        out[name] = reader.get_tensor(name)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tf-checkpoint", required=True, help="path to <name>_best.ckpt")
    ap.add_argument("--output", required=True, help="output .pt file")
    ap.add_argument("--scope", default="f0model", help="TF variable scope name")
    ap.add_argument("--name", default=None, help="model name recorded in metadata")
    args = ap.parse_args(argv)

    config = PiENetConfig()
    tf_vars = read_tf_checkpoint(args.tf_checkpoint)
    print(f"read {len(tf_vars)} variables from {args.tf_checkpoint}")

    state = build_state_dict(tf_vars, config, scope=args.scope)
    model = PiENet(config)
    model.load_state_dict(state, strict=True)
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        out,
        name=args.name or Path(args.tf_checkpoint).name.replace("_best.ckpt", ""),
        source=str(args.tf_checkpoint),
        converted_from="tensorflow-1 tf.train.Saver checkpoint",
        reference="Airaksinen et al., 'Data augmentation strategies for neural F0 "
        "estimation', ICASSP 2019",
    )
    print(
        f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{model.num_parameters():,} parameters)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
