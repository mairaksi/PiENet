import numpy as np
import pytest

from pienet.signal import (
    activations_to_f0,
    f0_bin_centers,
    f0_to_bin_index,
    f0_to_onehot,
    frame_signal,
    num_frames,
    resample,
)


def reference_get_frames(x, wl, hop):
    """The original sp_module.get_frames, kept here as the parity oracle."""
    n = int(np.ceil(x.shape[0] / hop))
    X = np.zeros((n, int(wl)), dtype=np.float32)
    pad = np.zeros(int(wl / 2), dtype=np.float32)
    x = np.concatenate((pad, x, pad))
    for i in range(n):
        X[i, :] = x[i * hop : i * hop + wl]
    return X


@pytest.mark.parametrize("length", [1, 160, 161, 1000, 16000, 16123])
def test_framing_matches_original(length):
    x = np.random.RandomState(0).randn(length).astype(np.float32)
    assert np.array_equal(frame_signal(x, 512, 160), reference_get_frames(x, 512, 160))


def test_num_frames():
    assert num_frames(0, 160) == 0
    assert num_frames(1, 160) == 1
    assert num_frames(160, 160) == 1
    assert num_frames(161, 160) == 2


def test_frame_empty_signal():
    assert frame_signal(np.zeros(0, dtype=np.float32), 512, 160).shape == (0, 512)


def test_bin_centers_span_range():
    centers = f0_bin_centers(50.0, 500.0, 351)
    assert centers.shape == (350,)
    assert centers[0] == pytest.approx(50.0, rel=1e-6)
    assert centers[-1] == pytest.approx(500.0, rel=1e-6)
    # log-spaced: constant ratio between neighbours
    ratios = centers[1:] / centers[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-5)


def test_f0_roundtrip_through_bins():
    centers = f0_bin_centers()
    idx = f0_to_bin_index(centers)
    assert np.array_equal(idx, np.arange(350))


def test_unvoiced_maps_to_last_bin():
    idx = f0_to_bin_index(np.array([0.0, -1.0, 120.0], dtype=np.float32))
    assert idx[0] == 350 and idx[1] == 350 and idx[2] < 350


def test_onehot_matches_indices():
    f0 = np.array([0.0, 100.0, 220.0, 480.0], dtype=np.float32)
    onehot = f0_to_onehot(f0)
    assert onehot.shape == (4, 351)
    assert np.array_equal(onehot.argmax(axis=1), f0_to_bin_index(f0))
    assert np.all(onehot.sum(axis=1) == 1.0)


def test_activations_to_f0_argmax():
    act = np.zeros((3, 351), dtype=np.float32)
    act[0, 0] = 1.0       # lowest pitch bin
    act[1, 349] = 1.0     # highest pitch bin
    act[2, 350] = 1.0     # unvoiced
    f0, voicing = activations_to_f0(act)
    assert f0[0] == pytest.approx(50.0, rel=1e-5)
    assert f0[1] == pytest.approx(500.0, rel=1e-5)
    assert f0[2] == 0.0
    assert voicing[2] == pytest.approx(0.0)


def test_voicing_threshold_overrides_argmax():
    act = np.zeros((1, 351), dtype=np.float32)
    act[0, 350] = 0.6     # arg-max says unvoiced
    act[0, 100] = 0.4
    assert activations_to_f0(act)[0][0] == 0.0
    f0, _ = activations_to_f0(act, voicing_threshold=0.3)
    assert f0[0] > 0.0


def test_interpolation_stays_between_neighbouring_bins():
    centers = f0_bin_centers()
    act = np.zeros((1, 351), dtype=np.float32)
    act[0, 99], act[0, 100], act[0, 101] = 0.2, 0.5, 0.3
    f0, _ = activations_to_f0(act, interpolate=True)
    assert centers[100] < f0[0] < centers[101]


def test_resample_changes_length():
    x = np.random.RandomState(0).randn(8000).astype(np.float32)
    y = resample(x, 8000, 16000)
    assert abs(y.shape[0] - 16000) <= 16
    assert resample(x, 16000, 16000) is not None
