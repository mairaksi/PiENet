import numpy as np
import pytest
import torch

from pienet.augmentation import (
    AugmentConfig,
    NoiseBank,
    add_channel_noise,
    add_noise,
    augment,
)
from pienet.config import PiENetConfig
from pienet.data import PAD_INDEX, F0Dataset, collate_batch, read_scp
from pienet.model import PiENet
from pienet.train import TrainConfig, Trainer, split_paths

CONFIG = PiENetConfig(
    win_length=32,
    hop_length=8,
    n_bins=11,
    residual_channels=8,
    postnet_channels=8,
    dilations=(1, 2),
)


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #

def test_add_noise_hits_the_requested_snr():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    y = add_noise(x, 10.0, rng)
    snr = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(y - x))
    assert snr == pytest.approx(10.0, abs=0.1)


def test_channel_noise_preserves_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000).astype(np.float32)
    assert add_channel_noise(x, rng).shape == x.shape


def test_augment_is_reproducible_and_non_destructive():
    x = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
    original = x.copy()
    a = augment(x, rng=np.random.default_rng(7))
    b = augment(x, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, original)


def test_augment_can_be_disabled():
    x = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
    cfg = AugmentConfig(channel_noise_prob=0.0, additive_noise_prob=0.0)
    np.testing.assert_array_equal(augment(x, config=cfg), x)


def test_noise_bank_loops_short_recordings():
    bank = NoiseBank([np.arange(10, dtype=np.float32)])
    drawn = bank.draw(35, np.random.default_rng(0))
    assert drawn.shape == (35,)


def test_augment_handles_silence():
    silence = np.zeros(1000, dtype=np.float32)
    out = augment(silence, config=AugmentConfig(channel_noise_prob=0.0))
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

@pytest.fixture
def toy_corpus(tmp_path):
    """Three synthetic utterances with matching reference F0 files."""
    import scipy.io.wavfile as wavfile

    wavs, f0s = [], []
    rng = np.random.default_rng(0)
    for i, n_samples in enumerate((800, 1200, 400)):
        t = np.arange(n_samples) / CONFIG.sample_rate
        f0_hz = 120.0 + 20 * i
        y = 0.3 * np.sin(2 * np.pi * f0_hz * t) + 0.01 * rng.standard_normal(n_samples)
        wav_path = tmp_path / f"u{i}.wav"
        wavfile.write(wav_path, CONFIG.sample_rate, (y * 32767).astype(np.int16))

        n_frames = int(np.ceil(n_samples / CONFIG.hop_length))
        f0 = np.full(n_frames, f0_hz, dtype=np.float32)
        f0[: n_frames // 4] = 0.0  # some unvoiced frames
        f0_path = tmp_path / f"u{i}.f0"
        f0.tofile(f0_path)

        wavs.append(str(wav_path))
        f0s.append(str(f0_path))

    (tmp_path / "wavs.scp").write_text("\n".join(wavs) + "\n")
    (tmp_path / "f0s.scp").write_text("\n".join(f0s) + "\n")
    return tmp_path, wavs, f0s


def test_read_scp(toy_corpus):
    tmp_path, wavs, _ = toy_corpus
    assert read_scp(tmp_path / "wavs.scp") == wavs


def test_dataset_item_shapes(toy_corpus):
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG)
    frames, targets = ds[0]
    assert frames.shape == (targets.shape[0], CONFIG.win_length)
    assert targets.dtype == torch.int64
    assert targets.max() <= CONFIG.n_bins - 1


def test_dataset_rejects_mismatched_f0_length(tmp_path, toy_corpus):
    _, wavs, f0s = toy_corpus
    bad = tmp_path / "bad.f0"
    np.zeros(3, dtype=np.float32).tofile(bad)
    with pytest.raises(ValueError, match="F0 values"):
        F0Dataset([wavs[0]], [str(bad)], CONFIG)[0]


def test_dataset_crops_to_max_frames(toy_corpus):
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG, max_frames=20)
    frames, targets = ds[1]
    assert frames.shape[0] == 20 and targets.shape[0] == 20


def test_collate_pads_and_masks(toy_corpus):
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG)
    batch = collate_batch([ds[i] for i in range(3)])
    assert batch.frames.shape[0] == 3
    assert batch.targets.shape[:2] == batch.frames.shape[:2]
    assert int(batch.lengths.max()) == batch.frames.shape[1]
    shortest = int(batch.lengths.argmin())
    assert (batch.targets[shortest, int(batch.lengths[shortest]) :] == PAD_INDEX).all()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def test_split_is_deterministic():
    wavs = [f"w{i}" for i in range(10)]
    f0s = [f"f{i}" for i in range(10)]
    a = split_paths(wavs, f0s, 0.2, seed=1)
    b = split_paths(wavs, f0s, 0.2, seed=1)
    assert a == b
    assert len(a[0]) == 8 and len(a[2]) == 2
    # pairs stay aligned
    assert [w[1:] for w in a[0]] == [f[1:] for f in a[1]]


def test_padded_frames_are_masked_out_of_the_loss(toy_corpus):
    """Padding must not contribute to the loss at all."""
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG, augment_config=None)
    model = PiENet(CONFIG).eval()
    trainer = Trainer(model, ds, config=TrainConfig(num_workers=0), device="cpu")

    batch = collate_batch([ds[0], ds[1]])
    with torch.inference_mode():
        logits = model(batch.frames)
        base = trainer._loss(logits, batch.targets)
        # scrambling the logits under the padding must not move the loss
        perturbed = logits.clone()
        shortest = int(batch.lengths.argmin())
        perturbed[shortest, int(batch.lengths[shortest]) :] += 100.0
        assert trainer._loss(perturbed, batch.targets).item() == pytest.approx(
            base.item(), rel=1e-6
        )


def test_batching_matches_single_away_from_the_padding(toy_corpus):
    """Interior frames are unaffected by how the batch was padded; frames
    within one receptive field of the boundary see the padding as context."""
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG, augment_config=None)
    model = PiENet(CONFIG).eval()

    single = collate_batch([ds[0]])
    padded = collate_batch([ds[0], ds[1]])
    n = single.frames.shape[1]
    rf = model.receptive_field_frames
    assert n > rf
    with torch.inference_mode():
        a = model(single.frames)[0, : n - rf]
        b = model(padded.frames)[0, : n - rf]
    assert torch.allclose(a, b, atol=1e-5)


def test_training_reduces_loss(tmp_path, toy_corpus):
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG, augment_config=None)
    model = PiENet(CONFIG)
    trainer = Trainer(
        model,
        ds,
        val_dataset=ds,
        config=TrainConfig(
            epochs=8,
            batch_size=3,
            learning_rate=3e-3,
            num_workers=0,
            output_dir=str(tmp_path / "out"),
            model_name="toy",
            log_every=0,
        ),
        device="cpu",
    )
    history = trainer.fit()
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert (tmp_path / "out" / "toy_best.pt").exists()
    assert (tmp_path / "out" / "toy_history.json").exists()
    assert "val_accuracy" in history[-1]


def test_resume_restores_state(tmp_path, toy_corpus):
    _, wavs, f0s = toy_corpus
    ds = F0Dataset(wavs, f0s, CONFIG, augment_config=None)
    cfg = TrainConfig(
        epochs=2, batch_size=3, num_workers=0,
        output_dir=str(tmp_path / "out"), model_name="toy", log_every=0,
    )
    Trainer(PiENet(CONFIG), ds, ds, cfg, device="cpu").fit()

    resumed = Trainer(PiENet(CONFIG), ds, ds, cfg, device="cpu")
    resumed.load_checkpoint(tmp_path / "out" / "toy_last.pt")
    assert resumed.start_epoch == 2
    assert len(resumed.history) == 2
