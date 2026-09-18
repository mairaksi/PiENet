from pathlib import Path

import numpy as np
import pytest

from pienet.cli import collect_inputs, main

REPO = Path(__file__).resolve().parents[1]
WAV = REPO / "wavs" / "arctic_a0001.wav"

pytestmark = pytest.mark.skipif(not WAV.exists(), reason="sample wav not available")


def test_collect_inputs_single_file():
    assert collect_inputs(str(WAV)) == [str(WAV)]


def test_collect_inputs_directory():
    assert str(WAV) in collect_inputs(str(WAV.parent))


def test_collect_inputs_scp(tmp_path):
    scp = tmp_path / "list.scp"
    scp.write_text(f"{WAV}\n\n{WAV}\n")
    assert collect_inputs(str(scp)) == [str(WAV), str(WAV)]


def test_collect_inputs_missing():
    with pytest.raises(SystemExit):
        collect_inputs("/nonexistent/path/xyz")


def test_estimate_writes_ascii(tmp_path):
    out = tmp_path / "f0"
    assert main(["estimate", str(WAV), str(out), "--device", "cpu", "-q"]) == 0
    written = out / "arctic_a0001.f0"
    assert written.exists()
    f0 = np.loadtxt(written)
    assert f0.ndim == 1 and (f0 >= 0).all() and (f0 < 600).all()


@pytest.mark.parametrize(
    "fmt,suffix", [("npy", ".npy"), ("csv", ".csv"), ("f32", ".f0")]
)
def test_estimate_formats(tmp_path, fmt, suffix):
    out = tmp_path / fmt
    assert (
        main(["estimate", str(WAV), str(out), "-f", fmt, "--device", "cpu", "-q"]) == 0
    )
    written = out / f"arctic_a0001{suffix}"
    assert written.exists() and written.stat().st_size > 0
    if fmt == "npy":
        assert np.load(written).ndim == 1
    if fmt == "f32":
        assert np.fromfile(written, dtype=np.float32).ndim == 1


def test_formats_agree(tmp_path):
    main(["estimate", str(WAV), str(tmp_path / "a"), "-f", "ascii", "-d", "cpu", "-q"])
    main(["estimate", str(WAV), str(tmp_path / "b"), "-f", "npy", "-d", "cpu", "-q"])
    ascii_f0 = np.loadtxt(tmp_path / "a" / "arctic_a0001.f0")
    npy_f0 = np.load(tmp_path / "b" / "arctic_a0001.npy")
    np.testing.assert_allclose(ascii_f0, npy_f0, atol=0.005)


def test_info_runs(capsys):
    assert main(["info"]) == 0
    assert "parameters" in capsys.readouterr().out


def test_bad_file_is_reported_not_fatal(tmp_path, capsys):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")
    scp = tmp_path / "list.scp"
    scp.write_text(f"{WAV}\n{broken}\n")
    assert main(["estimate", str(scp), str(tmp_path / "out"), "-d", "cpu", "-q"]) == 1
    assert (tmp_path / "out" / "arctic_a0001.f0").exists()
    assert "FAILED" in capsys.readouterr().err
