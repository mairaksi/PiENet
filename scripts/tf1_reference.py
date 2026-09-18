#!/usr/bin/env python3
"""Regenerate the TensorFlow 1 reference outputs used by the parity tests.

Runs the ORIGINAL TF1 graph (``legacy/model_fundf.py``) through
``tensorflow.compat.v1`` under a modern TensorFlow 2 install, restores the
original checkpoint, and stores the logits and decoded F0 contour for the
sample utterance. ``tests/test_inference.py`` asserts that the PyTorch port
reproduces them.

This script is only needed when re-deriving the reference; the resulting
``.npz`` is committed, so the test suite does not require TensorFlow.

Usage::

    pip install "tensorflow-cpu>=2.12"
    python scripts/tf1_reference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def install_tf1_shims():
    """Make the 2019-era code importable under TensorFlow 2."""
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    class _Layers:
        @staticmethod
        def xavier_initializer_conv2d():
            return tf.glorot_uniform_initializer()

    class _Contrib:
        layers = _Layers()

    class _TFLayers:
        @staticmethod
        def dropout(inputs, rate=0.5, training=False):
            return inputs if training is False else tf.nn.dropout(inputs, rate=rate)

    tf.contrib = _Contrib()          # gone in TF2
    tf.layers = _TFLayers()          # gone with Keras 3
    sys.modules["tensorflow"] = tf   # `import tensorflow as tf` -> compat.v1
    return tf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--legacy-dir", default=str(REPO / "legacy"))
    ap.add_argument(
        "--tf-checkpoint", default=str(REPO / "legacy/saved_models/gtaug_best.ckpt")
    )
    ap.add_argument("--wav", default=str(REPO / "wavs/arctic_a0001.wav"))
    ap.add_argument(
        "--output",
        default=str(REPO / "tests/data/tf1_reference_arctic_a0001.npz"),
    )
    args = ap.parse_args(argv)

    tf = install_tf1_shims()
    sys.path.insert(0, args.legacy_dir)

    import model_fundf as model  # noqa: E402  (needs the shims first)
    import scipy.io.wavfile as wavfile
    import sp_module as sp  # noqa: E402

    winlen, hop, n_bins = 512, 160, 351
    net = model.CNET(
        name="f0model",
        input_channels=winlen,
        output_channels=n_bins,
        dilations=[1, 2, 4, 8, 1, 2, 4, 8],
        filter_width=5,
        residual_channels=128,
        postnet_channels=256,
    )
    x = tf.placeholder(shape=(None, None, winlen), dtype=tf.float32)
    logits = net.forward_pass(x)
    probs = tf.nn.softmax(logits)

    with tf.Session() as sess:
        tf.train.Saver().restore(sess, args.tf_checkpoint)
        _, y = wavfile.read(args.wav)
        y = np.float32(y / (2**15))
        frames = np.reshape(sp.get_frames(y, winlen, hop), [1, -1, winlen])
        lg, pr = sess.run([logits, probs], feed_dict={x: frames})

    f0 = sp.getF0fromActivations(np.reshape(pr, [-1, n_bins]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, f0=f0, logits=lg.astype(np.float32))
    print(f"wrote {out}: {f0.shape[0]} frames, {int((f0 > 0).sum())} voiced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
