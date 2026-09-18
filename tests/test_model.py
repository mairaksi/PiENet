import pytest
import torch

from pienet import PiENet, PiENetConfig
from pienet.model import available_checkpoints


@pytest.fixture(scope="module")
def small_config():
    return PiENetConfig(
        win_length=32,
        hop_length=8,
        n_bins=11,
        residual_channels=8,
        postnet_channels=8,
        dilations=(1, 2),
    )


def test_config_rejects_even_filter_width():
    with pytest.raises(ValueError):
        PiENetConfig(filter_width=4)


def test_config_roundtrip(small_config):
    assert PiENetConfig.from_dict(small_config.to_dict()) == small_config


def test_receptive_field():
    c = PiENetConfig()
    # 2 * (1+2+4+8) * 2 dilated layers + 2 postnet convolutions
    assert c.receptive_field_frames == 64


def test_forward_shapes(small_config):
    model = PiENet(small_config).eval()
    x = torch.randn(3, 17, small_config.win_length)
    assert model(x).shape == (3, 17, small_config.n_bins)
    assert model(x[0]).shape == (17, small_config.n_bins)


def test_forward_rejects_wrong_window(small_config):
    model = PiENet(small_config).eval()
    with pytest.raises(ValueError):
        model(torch.randn(1, 5, small_config.win_length + 1))


def test_batching_is_equivalent_to_single(small_config):
    model = PiENet(small_config).eval()
    x = torch.randn(4, 23, small_config.win_length)
    with torch.inference_mode():
        batched = model(x)
        single = torch.stack([model(x[i : i + 1])[0] for i in range(4)])
    assert torch.allclose(batched, single, atol=1e-5)


def test_time_shift_equivariance(small_config):
    """Symmetric padding: an interior frame's output must not depend on
    how much silence is appended after it."""
    model = PiENet(small_config).eval()
    x = torch.randn(1, 60, small_config.win_length)
    rf = model.receptive_field_frames
    with torch.inference_mode():
        full = model(x)
        cropped = model(x[:, : 30 + rf])
    assert torch.allclose(full[:, :30], cropped[:, :30], atol=1e-5)


def test_zero_biases_after_init(small_config):
    model = PiENet(small_config)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            assert torch.all(module.bias == 0)


def test_dropout_only_active_in_training(small_config):
    torch.manual_seed(0)
    model = PiENet(small_config)
    x = torch.randn(1, 10, small_config.win_length)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(x), model(x))
    model.train()
    assert not torch.allclose(model(x), model(x))


def test_save_and_load(tmp_path, small_config):
    model = PiENet(small_config).eval()
    path = tmp_path / "m.pt"
    model.save_pretrained(path, note="unit test")
    loaded = PiENet.from_pretrained(path)
    assert loaded.config == small_config
    x = torch.randn(1, 12, small_config.win_length)
    with torch.inference_mode():
        assert torch.allclose(model(x), loaded(x))


def test_from_pretrained_rejects_unknown_name():
    with pytest.raises(FileNotFoundError):
        PiENet.from_pretrained("no-such-model")


def test_bundled_checkpoint_present():
    assert "gtaug" in available_checkpoints()


def test_pretrained_model_shape():
    model = PiENet.from_pretrained()
    assert model.num_parameters() == 2_256_351
    assert model.config.sample_rate == 16000
    assert not model.training
