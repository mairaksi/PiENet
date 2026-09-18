"""Command line interface: ``pienet estimate`` and ``pienet train``."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import __version__

__all__ = ["main", "collect_inputs"]

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".w64")


def collect_inputs(source: str) -> List[str]:
    """Resolve a CLI input into a list of audio files.

    ``source`` may be a single audio file, a directory of audio files, or a
    text/``.scp`` file listing one path per line.
    """
    p = Path(source)
    if p.is_dir():
        files = sorted(
            str(f) for f in p.iterdir() if f.suffix.lower() in AUDIO_SUFFIXES
        )
        if not files:
            raise SystemExit(f"no audio files found in directory {source!r}")
        return files
    if not p.exists():
        matches = sorted(glob.glob(source))
        if matches:
            return matches
        raise SystemExit(f"input {source!r} does not exist")
    if p.suffix.lower() in AUDIO_SUFFIXES:
        return [str(p)]
    with open(p) as fh:
        files = [line.strip() for line in fh if line.strip()]
    if not files:
        raise SystemExit(f"file list {source!r} is empty")
    return files


def _write_f0(
    f0: np.ndarray,
    voicing: Optional[np.ndarray],
    path: Path,
    fmt: str,
    hop_seconds: float,
) -> None:
    if fmt == "ascii":
        np.savetxt(path, f0, fmt="%.2f")
    elif fmt == "f32":
        f0.astype(np.float32).tofile(path)
    elif fmt == "npy":
        np.save(path, f0)
    elif fmt == "csv":
        times = np.arange(f0.shape[0], dtype=np.float64) * hop_seconds
        if voicing is None:
            data = np.column_stack([times, f0])
            header = "time,f0"
        else:
            data = np.column_stack([times, f0, voicing])
            header = "time,f0,voicing"
        np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"unknown format {fmt!r}")


_SUFFIX = {"ascii": ".f0", "f32": ".f0", "npy": ".npy", "csv": ".csv"}


def cmd_estimate(args: argparse.Namespace) -> int:
    from .inference import F0Estimator

    files = collect_inputs(args.input)
    target_dir = Path(args.output)
    target_dir.mkdir(parents=True, exist_ok=True)

    estimator = F0Estimator(
        model=args.model, device=args.device, chunk_frames=args.chunk_frames
    )
    hop_seconds = estimator.config.hop_length / estimator.config.sample_rate
    suffix = args.suffix or _SUFFIX[args.format]

    print(
        f"loaded {args.model} on {estimator.device}; "
        f"processing {len(files)} file(s)",
        file=sys.stderr,
    )

    failures = 0
    for i, wav in enumerate(files, 1):
        try:
            f0, voicing = estimator.estimate_file(
                wav,
                return_voicing=True,
                voicing_threshold=args.voicing_threshold,
                interpolate=args.interpolate,
            )
        except Exception as exc:  # keep going through a bad file in a long list
            failures += 1
            print(f"[{i}/{len(files)}] {wav}: FAILED ({exc})", file=sys.stderr)
            continue

        out = target_dir / (Path(wav).stem + suffix)
        _write_f0(f0, voicing, out, args.format, hop_seconds)
        if not args.quiet:
            voiced = int((f0 > 0).sum())
            print(
                f"[{i}/{len(files)}] {wav} -> {out} "
                f"({f0.shape[0]} frames, {voiced} voiced)",
                file=sys.stderr,
            )

    if failures:
        print(f"{failures} file(s) failed", file=sys.stderr)
    return 1 if failures else 0


def cmd_train(args: argparse.Namespace) -> int:
    from .augmentation import AugmentConfig
    from .config import PiENetConfig
    from .train import TrainConfig, train_from_scp

    augment_config = AugmentConfig()
    if args.no_augment:
        augment_config = AugmentConfig(
            channel_noise_prob=0.0, additive_noise_prob=0.0, random_gain=False
        )

    train_from_scp(
        wav_scp=args.wav_scp,
        f0_scp=args.f0_scp,
        noise_scp=args.noise_scp,
        model_config=PiENetConfig(),
        train_config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_frames=args.max_frames,
            num_workers=args.num_workers,
            seed=args.seed,
            amp=args.amp,
            output_dir=args.output_dir,
            model_name=args.name,
        ),
        augment_config=augment_config,
        f0_downsample=2 if args.downsample_f0 else 1,
        val_fraction=args.val_fraction,
        resume=args.resume,
        init_from=args.init_from,
        device=args.device,
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import torch

    from .model import PiENet, available_checkpoints

    print(f"pienet {__version__}  (torch {torch.__version__})")
    print(f"bundled checkpoints: {', '.join(available_checkpoints()) or 'none'}")
    print(f"cuda available: {torch.cuda.is_available()}")
    model = PiENet.from_pretrained(args.model)
    c = model.config
    print(f"\nmodel: {args.model}")
    print(f"  parameters      : {model.num_parameters():,}")
    print(f"  sample rate     : {c.sample_rate} Hz")
    print(f"  window / hop    : {c.win_length} / {c.hop_length} samples "
          f"({1000 * c.hop_length / c.sample_rate:.1f} ms frame shift)")
    print(f"  F0 range        : {c.f0_min:g}-{c.f0_max:g} Hz in {c.n_pitch_bins} bins")
    print(f"  receptive field : +/- {c.receptive_field_frames} frames")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pienet",
        description="Noise-robust neural F0 (pitch) estimation for speech.",
    )
    parser.add_argument("--version", action="version", version=f"pienet {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------- estimate ---------------- #
    est = sub.add_parser(
        "estimate",
        help="estimate F0 for audio files",
        description="Estimate F0 for an audio file, a directory of audio "
        "files, or a list of files given in a .scp/.txt file.",
    )
    est.add_argument(
        "input",
        nargs="?",
        default="test_wavs.scp",
        help="audio file, directory, or file list (default: test_wavs.scp)",
    )
    est.add_argument(
        "output", nargs="?", default="f0", help="output directory (default: f0/)"
    )
    est.add_argument("-m", "--model", default="gtaug", help="checkpoint name or path")
    est.add_argument("-d", "--device", default=None, help="cpu, cuda, mps, ...")
    est.add_argument(
        "-f",
        "--format",
        default="ascii",
        choices=("ascii", "f32", "npy", "csv"),
        help="output format (default: ascii, one value per line)",
    )
    est.add_argument("--suffix", default=None, help="override the output suffix")
    est.add_argument(
        "--voicing-threshold",
        type=float,
        default=None,
        help="voice a frame when 1 - p(unvoiced) >= this value "
        "(default: arg-max, as in the original implementation)",
    )
    est.add_argument(
        "--interpolate",
        action="store_true",
        help="parabolic sub-bin refinement of the F0 estimate",
    )
    est.add_argument(
        "--chunk-frames",
        type=int,
        default=20000,
        help="max frames processed at once (default: 20000)",
    )
    est.add_argument("-q", "--quiet", action="store_true")
    est.set_defaults(func=cmd_estimate)

    # ---------------- train ---------------- #
    tr = sub.add_parser("train", help="train a model from .scp file lists")
    tr.add_argument("--wav-scp", default="train_wavs.scp")
    tr.add_argument("--f0-scp", default="train_f0s.scp")
    tr.add_argument(
        "--noise-scp",
        default=None,
        help="noise recordings for additive augmentation (default: white noise)",
    )
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--learning-rate", type=float, default=1e-4)
    tr.add_argument("--max-frames", type=int, default=1500)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--val-fraction", type=float, default=0.1)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    tr.add_argument("--no-augment", action="store_true")
    tr.add_argument(
        "--downsample-f0",
        action="store_true",
        help="reference F0 was computed with a 5 ms hop",
    )
    tr.add_argument("--output-dir", default="saved_models")
    tr.add_argument("--name", default="pienet")
    tr.add_argument("--resume", default=None, help="resume from *_last.pt")
    tr.add_argument("--init-from", default=None, help="fine-tune from a model .pt")
    tr.add_argument("-d", "--device", default=None)
    tr.set_defaults(func=cmd_train)

    # ---------------- info ---------------- #
    info = sub.add_parser("info", help="show version, devices and model details")
    info.add_argument("-m", "--model", default="gtaug")
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
