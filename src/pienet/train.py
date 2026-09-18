"""Training loop for PiENet.

Port of the TensorFlow 1 ``train.py``: Adam on a softmax cross-entropy over
F0 bins, with random noise augmentation, validation after every epoch and a
"best so far" checkpoint.

What changed beyond the framework swap:

* Minibatching. The original fed one utterance at a time. Padded frames are
  masked out of the loss (``ignore_index``), so they contribute no gradient.
  Note that frames within one receptive field (64 frames) of the padding
  boundary still see the zero padding as context, exactly as the frames at the
  true end of a recording do; set ``max_frames`` or batch similar lengths
  together if you want to keep that edge region small.
* Input dropout is now actually active during training. The original built the
  graph with ``training=False``, so its dropout layer was a no-op.
* Validation reports frame accuracy and gross pitch error alongside the loss.
* Checkpoints store the model config, so inference needs no matching code.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentation import AugmentConfig, NoiseBank
from .config import PiENetConfig
from .data import PAD_INDEX, F0Dataset, collate_batch, read_scp
from .model import PiENet
from .signal import f0_bin_centers

__all__ = ["TrainConfig", "Trainer", "train_from_scp", "split_paths"]

PathLike = Union[str, "os.PathLike[str]"]


@dataclass
class TrainConfig:
    """Optimisation and bookkeeping settings."""

    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 5.0
    max_frames: Optional[int] = 1500
    num_workers: int = 4
    seed: int = 42
    amp: bool = False
    output_dir: str = "saved_models"
    model_name: str = "pienet"
    log_every: int = 20


def split_paths(
    wavs: Sequence[str], f0s: Sequence[str], val_fraction: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Deterministic train/validation split of paired file lists."""
    n_val = int(round(val_fraction * len(wavs)))
    order = np.random.RandomState(seed=seed).permutation(len(wavs))
    wavs_a = np.asarray(wavs)[order]
    f0s_a = np.asarray(f0s)[order]
    return (
        list(wavs_a[n_val:]),
        list(f0s_a[n_val:]),
        list(wavs_a[:n_val]),
        list(f0s_a[:n_val]),
    )


class Trainer:
    """Trains a :class:`~pienet.model.PiENet` on paired audio/F0 data."""

    def __init__(
        self,
        model: PiENet,
        train_dataset: F0Dataset,
        val_dataset: Optional[F0Dataset] = None,
        config: Optional[TrainConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(self.config.seed)

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.config.amp and self.device.type == "cuda"
        )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, float]] = []
        self.best_val = math.inf
        self.start_epoch = 0

        centers = f0_bin_centers(
            model.config.f0_min, model.config.f0_max, model.config.n_bins
        )
        self._centers = torch.from_numpy(centers).to(self.device)

    # ------------------------------------------------------------------ #

    def _loader(self, dataset: F0Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_batch,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            persistent_workers=self.config.num_workers > 0,
        )

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=PAD_INDEX,
        )

    @torch.no_grad()
    def _metrics(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Frame accuracy, voicing error and gross pitch error (>20%)."""
        unvoiced = self.model.config.unvoiced_index
        mask = targets != PAD_INDEX
        if mask.sum() == 0:
            return {}
        pred = logits.argmax(dim=-1)[mask]
        ref = targets[mask]

        acc = (pred == ref).float().mean().item()
        voicing_err = ((pred == unvoiced) != (ref == unvoiced)).float().mean().item()

        both_voiced = (pred < unvoiced) & (ref < unvoiced)
        if both_voiced.any():
            pf = self._centers[pred[both_voiced].clamp(max=unvoiced - 1)]
            rf = self._centers[ref[both_voiced].clamp(max=unvoiced - 1)]
            gpe = ((pf - rf).abs() > 0.2 * rf).float().mean().item()
        else:
            gpe = float("nan")
        return {"accuracy": acc, "voicing_error": voicing_err, "gross_pitch_error": gpe}

    # ------------------------------------------------------------------ #

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.train_dataset.set_epoch(epoch)
        loader = self._loader(self.train_dataset, shuffle=True)

        total, n_batches = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                self.device.type, enabled=self.scaler.is_enabled()
            ):
                logits = self.model(batch.frames)
                loss = self._loss(logits, batch.targets)

            self.scaler.scale(loss).backward()
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item()
            n_batches += 1
            if self.config.log_every and step % self.config.log_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(loader)} "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        print(f"  epoch {epoch} took {time.time() - t0:.1f}s", flush=True)
        return total / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_dataset is None:
            return {}
        self.model.eval()
        loader = self._loader(self.val_dataset, shuffle=False)

        agg: Dict[str, List[float]] = {}
        total, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch.frames)
            total += self._loss(logits, batch.targets).item()
            n_batches += 1
            for k, v in self._metrics(logits, batch.targets).items():
                agg.setdefault(k, []).append(v)

        out = {"loss": total / max(n_batches, 1)}
        out.update({k: float(np.nanmean(v)) for k, v in agg.items()})
        return out

    # ------------------------------------------------------------------ #

    def fit(self) -> List[Dict[str, float]]:
        """Run the full training schedule, checkpointing after each epoch."""
        name = self.config.model_name
        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"Training epoch {epoch}", flush=True)
            train_loss = self.train_epoch(epoch)
            val = self.validate()

            record = {"epoch": epoch, "train_loss": train_loss}
            record.update({f"val_{k}": v for k, v in val.items()})
            self.history.append(record)
            print(
                "  "
                + "  ".join(f"{k}={v:.4f}" for k, v in record.items() if k != "epoch"),
                flush=True,
            )

            self.save_checkpoint(self.output_dir / f"{name}_last.pt", epoch)
            self.model.save_pretrained(self.output_dir / f"{name}.pt", **record)

            score = val.get("loss", train_loss)
            if score < self.best_val:
                self.best_val = score
                self.model.save_pretrained(
                    self.output_dir / f"{name}_best.pt", **record
                )
                print(f"  new best ({score:.4f}) -> {name}_best.pt", flush=True)

            with open(self.output_dir / f"{name}_history.json", "w") as fh:
                json.dump(self.history, fh, indent=2)

        return self.history

    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: PathLike, epoch: int) -> None:
        """Save a resumable training state (model + optimizer + history)."""
        torch.save(
            {
                "format": "pienet-train-v1",
                "config": self.model.config.to_dict(),
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "best_val": self.best_val,
                "history": self.history,
            },
            str(path),
        )

    def load_checkpoint(self, path: PathLike) -> None:
        """Resume from a checkpoint written by :meth:`save_checkpoint`."""
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])
        self.history = payload.get("history", [])
        self.best_val = payload.get("best_val", math.inf)
        self.start_epoch = int(payload.get("epoch", -1)) + 1
        print(f"resumed from {path} at epoch {self.start_epoch}", flush=True)


def train_from_scp(
    wav_scp: PathLike = "train_wavs.scp",
    f0_scp: PathLike = "train_f0s.scp",
    noise_scp: Optional[PathLike] = None,
    model_config: Optional[PiENetConfig] = None,
    train_config: Optional[TrainConfig] = None,
    augment_config: Optional[AugmentConfig] = None,
    f0_downsample: int = 1,
    val_fraction: float = 0.1,
    resume: Optional[PathLike] = None,
    init_from: Optional[PathLike] = None,
    device: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Train a model from Kaldi-style ``.scp`` file lists.

    Args:
        wav_scp: List of training audio files, one path per line.
        f0_scp: Matching list of reference F0 files (raw float32, 0 = unvoiced).
        noise_scp: Optional list of noise recordings for additive augmentation.
            ``None`` uses white noise.
        model_config: Network/signal configuration.
        train_config: Optimisation settings.
        augment_config: Augmentation settings; pass ``AugmentConfig(
            channel_noise_prob=0, additive_noise_prob=0)`` to disable.
        f0_downsample: Set to 2 when the reference F0 uses a 5 ms hop.
        val_fraction: Fraction of utterances held out for validation.
        resume: Resume training from a ``*_last.pt`` training checkpoint.
        init_from: Initialise weights from an existing model checkpoint
            (fine-tuning) without restoring the optimizer state.
        device: Torch device; ``None`` auto-selects CUDA when available.

    Returns:
        The per-epoch history records.
    """
    mcfg = model_config or PiENetConfig()
    tcfg = train_config or TrainConfig()

    wavs, f0s = read_scp(wav_scp), read_scp(f0_scp)
    tr_w, tr_f, va_w, va_f = split_paths(wavs, f0s, val_fraction, tcfg.seed)
    print(f"{len(tr_w)} training and {len(va_w)} validation utterances", flush=True)

    noise_bank = (
        NoiseBank.from_scp(noise_scp, mcfg.sample_rate) if noise_scp else None
    )
    acfg = augment_config or AugmentConfig()

    train_ds = F0Dataset(
        tr_w, tr_f, mcfg, acfg, noise_bank, f0_downsample, tcfg.max_frames, tcfg.seed
    )
    val_ds = (
        F0Dataset(va_w, va_f, mcfg, None, None, f0_downsample, None, tcfg.seed)
        if va_w
        else None
    )

    model = (
        PiENet.from_pretrained(init_from, map_location="cpu")
        if init_from
        else PiENet(mcfg)
    )
    trainer = Trainer(model, train_ds, val_ds, tcfg, device)
    if resume:
        trainer.load_checkpoint(resume)
    return trainer.fit()
