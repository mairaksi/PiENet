"""Inference tests, including numerical parity with the TensorFlow 1 model.

``tests/data/tf1_reference_arctic_a0001.npz`` holds the F0 contour and the
class logits produced by the original TensorFlow 1 graph for
``wavs/arctic_a0001.wav`` (generated once with
``scripts/tf1_reference.py``). The port must reproduce them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pienet import F0Estimator, PiENet, estimate_f0
from pienet.signal import frame_signal, read_audio

REPO = Path(__file__).resolve().parents[1]
WAV = REPO / "wavs" / "arctic_a0001.wav"
REFERENCE = Path(__file__).parent / "data" / "tf1_reference_arctic_a0001.npz"

pytestmark = pytest.mark.skipif(
    not WAV.exists(), reason="sample wav not available"
)


@pytest.fixture(scope="module")
def estimator():
    return F0Estimator(device="cpu")


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip("TF1 reference outputs not bundled")
    return np.load(REFERENCE)


def test_f0_matches_tensorflow_exactly(estimator, reference):
    y, sr = read_audio(WAV, target_sr=16000)
    f0 = estimator.estimate(y, sample_rate=sr)
    assert f0.shape == reference["f0"].shape
    np.testing.assert_array_equal(f0, reference["f0"])


def test_logits_match_tensorflow(reference):
    model = PiENet.from_pretrained()
    y, _ = read_audio(WAV, target_sr=16000)
    frames = frame_signal(y, 512, 160)
    with torch.inference_mode():
        logits = model(torch.from_numpy(frames)).numpy()
    ref = reference["logits"]
    # float32 accumulation order differs between the two frameworks
    assert np.abs(logits - ref).max() < 1e-2
    np.testing.assert_array_equal(logits.argmax(axis=1), ref.argmax(axis=1))


def test_chunking_does_not_change_the_result(reference):
    y, sr = read_audio(WAV, target_sr=16000)
    small = F0Estimator(device="cpu", chunk_frames=150)
    np.testing.assert_array_equal(small.estimate(y, sr), reference["f0"])


def test_chunk_frames_must_exceed_receptive_field():
    with pytest.raises(ValueError):
        F0Estimator(device="cpu", chunk_frames=100)


def test_frame_count_follows_hop(estimator):
    y, sr = read_audio(WAV, target_sr=16000)
    f0 = estimator.estimate(y, sample_rate=sr)
    assert f0.shape[0] == int(np.ceil(y.shape[0] / 160))


def test_voicing_is_returned(estimator):
    y, sr = read_audio(WAV, target_sr=16000)
    f0, voicing = estimator.estimate(y, sr, return_voicing=True)
    assert voicing.shape == f0.shape
    assert ((voicing >= 0.0) & (voicing <= 1.0)).all()
    # voiced frames should be confident
    assert voicing[f0 > 0].mean() > 0.5


def test_resampling_path_gives_similar_f0(estimator):
    import scipy.signal

    y, _ = read_audio(WAV, target_sr=16000)
    y8 = scipy.signal.resample_poly(y, 1, 2).astype(np.float32)
    f0_16 = estimator.estimate(y, 16000)
    f0_8 = estimator.estimate(y8, 8000)
    both = (f0_16 > 0) & (f0_8 > 0)
    assert both.sum() > 0.5 * (f0_16 > 0).sum()
    rel = np.abs(f0_8[both] - f0_16[both]) / f0_16[both]
    assert np.median(rel) < 0.05


def test_estimate_file_and_convenience_api(estimator):
    from_file = estimator.estimate_file(WAV)
    one_shot = estimate_f0(WAV, device="cpu")
    np.testing.assert_array_equal(from_file, one_shot)


def test_silence_is_unvoiced(estimator):
    f0 = estimator.estimate(np.zeros(16000, dtype=np.float32), 16000)
    assert (f0 == 0).mean() > 0.9


def test_short_input_does_not_crash(estimator):
    f0 = estimator.estimate(np.random.RandomState(0).randn(200) * 0.01, 16000)
    assert f0.shape[0] == 2
