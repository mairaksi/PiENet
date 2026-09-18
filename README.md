# PiENet

Pitch estimation network (PiENet) for noise-robust neural F0 estimation of speech signals.

The best performing model from [1] (`GTE-AUG`) is supplied as a pre-trained model. It was trained using additive and convolutional noise augmentation as well as vocoder-based ground truth enhancement (see the publication for details).

> **Version 2.0 is a PyTorch port.** The original TensorFlow 1 implementation is preserved under [`legacy/`](legacy/). The pre-trained weights were converted, not retrained: the PyTorch model reproduces the TensorFlow model's F0 output exactly on the bundled test utterance (see [Verification](#verification)).

Article: [IEEE Xplore](https://ieeexplore.ieee.org/document/8683041) · [ResearchGate](https://www.researchgate.net/publication/331012502_Data_Augmentation_Strategies_for_Neural_Network_F0_Estimation)

## Installation

```bash
pip install -e .                # core: torch, numpy, scipy
pip install -e ".[audio]"       # + soundfile, for flac/ogg/non-PCM wav input
pip install -e ".[dev]"         # + pytest, ruff
```

Python 3.9 or newer, PyTorch 2.3 or newer. CPU, CUDA and Apple MPS all work; the device is selected automatically.

## Usage

### Command line

```bash
pienet estimate speech.wav                    # -> f0/speech.f0
pienet estimate recordings/ out/              # every audio file in a directory
pienet estimate files.scp out/ -f csv         # a list of files, CSV output
pienet estimate speech.wav out/ --device cpu
pienet info                                   # model, devices, version
```

Options worth knowing:

| Flag | Effect |
| --- | --- |
| `-f, --format` | `ascii` (default, one value per line), `f32` (raw float32), `npy`, `csv` (time, f0, voicing) |
| `--voicing-threshold` | Voice a frame when `1 - p(unvoiced) >= T` instead of taking the arg-max. Lets you trade voiced/unvoiced errors without retraining. |
| `--interpolate` | Parabolic sub-bin refinement — finer than the 350-bin grid (~0.66 % steps). |
| `--chunk-frames` | Bound memory on very long recordings. Chunking is exact, not approximate. |
| `--device` | `cpu`, `cuda`, `mps`, … |

The original script interface still works:

```bash
python generate.py                            # test_wavs.scp -> f0/
python generate.py input.wav
python generate.py input_list.scp target_dir
python generate.py input_dir target_dir
```

### Python

```python
import pienet

f0 = pienet.estimate_f0("speech.wav")         # Hz per 10 ms frame, 0.0 = unvoiced

# Reuse the model across many files
est = pienet.F0Estimator(device="cuda")
f0, voicing = est.estimate_file("speech.wav", return_voicing=True)

# Work from an in-memory waveform at any sample rate
import soundfile as sf
y, sr = sf.read("speech.flac")
f0 = est.estimate(y, sample_rate=sr)
```

The model itself is a plain `torch.nn.Module`:

```python
import torch
from pienet import PiENet
from pienet.signal import frame_signal, activations_to_f0

model = PiENet.from_pretrained("gtaug").eval()
frames = frame_signal(y, 512, 160)                        # (n_frames, 512)
with torch.inference_mode():
    logits = model(torch.from_numpy(frames))              # (n_frames, 351)
f0, voicing = activations_to_f0(torch.softmax(logits, -1).numpy())
```

## The model

| | |
| --- | --- |
| Input | Framed raw waveform, 512-sample (32 ms) window, 160-sample (10 ms) hop |
| Trunk | 1×1 input projection → 8 gated dilated convolution blocks (width 5, dilations 1-2-4-8-1-2-4-8, 128 channels) with residual and skip connections |
| Output | Two width-5 convolutions → softmax over 351 classes: 350 log-spaced F0 bins from 50 to 500 Hz, plus one "unvoiced" class |
| Size | 2,256,351 parameters (9 MB) |
| Context | ±64 frames (±640 ms), non-causal |

Frames follow the convention `n_frames = ceil(n_samples / hop)`, with the signal zero-padded by `win_length / 2` at both ends.

## Training your own model

```bash
pienet train --wav-scp train_wavs.scp --f0-scp train_f0s.scp \
             --epochs 100 --batch-size 8 --amp --name mymodel
```

* `train_wavs.scp` lists the training audio, one path per line; `train_f0s.scp` lists the matching reference F0 files (raw `float32`, one value per frame, `0` for unvoiced).
* The number of F0 values must match the framing convention above. If your reference was computed with a 5 ms hop, pass `--downsample-f0`.
* `--noise-scp noiselist.scp` samples additive noise from a corpus instead of using white noise.
* `--no-augment` disables augmentation; `--resume saved_models/mymodel_last.pt` continues an interrupted run; `--init-from` fine-tunes from an existing model.

Checkpoints land in `saved_models/`: `<name>_best.pt` (lowest validation loss), `<name>_last.pt` (resumable state) and `<name>_history.json` (per-epoch loss, frame accuracy, voicing error and gross pitch error).

From Python:

```python
from pienet.train import train_from_scp, TrainConfig

train_from_scp(
    "train_wavs.scp", "train_f0s.scp",
    train_config=TrainConfig(epochs=100, batch_size=8, amp=True),
)
```

For vocoder-based augmentation you need the vocoder of your choice — the [GlottDNN](https://github.com/ljuvela/GlottDNN) vocoder was used in [1]. For ground truth enhancement, process the training wavs offline with the vocoder. The code for diversity augmentation is not part of this release.

## Verification

`tests/data/tf1_reference_arctic_a0001.npz` holds the class logits and decoded F0 contour produced by the *original* TensorFlow 1 graph. The test suite asserts the port reproduces them:

```bash
pytest                    # 66 tests, no TensorFlow required
```

On the bundled utterance the decoded F0 contour is **identical**, frame for frame, and the raw logits agree to 4.9e-4 absolute (7e-7 relative) — float32 accumulation-order noise between the two frameworks.

To re-derive the conversion yourself:

```bash
pip install "tensorflow-cpu>=2.12"
python scripts/tf1_reference.py                     # regenerate the reference
python scripts/convert_tf_checkpoint.py \
    --tf-checkpoint legacy/saved_models/gtaug_best.ckpt \
    --output src/pienet/assets/gtaug.pt
```

## What changed in the port

Behaviour-preserving by default; the differences are deliberate and listed here.

**Same results**

* Framing, log-spaced bin layout, arg-max decoding and the unvoiced class are bit-for-bit identical to the original.
* Convolutions use symmetric padding, which is what TensorFlow's `'SAME'` does for odd kernels. Long-file chunking overlaps by the full receptive field, so it does not change the output.

**Fixed**

* Integer PCM is now normalised by its own full scale. The original divided everything by `2**15`, so 24- and 32-bit files came out thousands of times too loud.
* Input dropout is now actually applied during training. The original built the training graph with `training=False`, so its dropout layer was a no-op — the published model was effectively trained without it.
* `add_noise_file` modified the caller's waveform in place and used `noise_samples == None` for its check; augmentation now returns a new array.
* A noise recording shorter than the utterance raised in `np.random.randint`; it is now looped.
* Resampling uses a polyphase filter (`resample_poly`) rather than `scipy.signal.resample`, avoiding the circular-convolution artefacts of the Fourier method at signal edges.

**New**

* Minibatched training with length padding masked out of the loss, gradient clipping, mixed precision, resumable checkpoints, and validation metrics (frame accuracy, voicing error, gross pitch error) instead of loss alone.
* `--voicing-threshold` and `--interpolate` decoding options.
* `csv`, `npy` and raw-`float32` output formats.
* Checkpoints carry their own config, so a model file is self-describing.
* An importable Python API and a `pienet` console command.

**Removed**

* `tf.contrib`, `tf.placeholder`, `tf.Session` and the TF1 `Saver` checkpoint format. The old files remain in `legacy/` for reference and for re-running the conversion.

## Repository layout

```
src/pienet/            the package
  config.py            model and signal configuration
  model.py             PiENet nn.Module
  signal.py            framing, F0 bins, audio I/O
  inference.py         F0Estimator
  data.py              dataset and batching
  augmentation.py      noise augmentation
  train.py             training loop
  cli.py               pienet command
  assets/gtaug.pt      converted pre-trained model
scripts/               TF1 checkpoint conversion, reference generation
tests/                 test suite incl. TensorFlow parity
legacy/                the original TensorFlow 1 implementation
generate.py, train.py  wrappers for the original command lines
```

## Licence

Distributed under the Apache 2.0 license. See [LICENSE](LICENSE) for further details.

## Reference

[1] M. Airaksinen, L. Juvela, P. Alku and O. Räsänen: "Data augmentation strategies for neural F0 estimation", Proc. ICASSP 2019.

Available: [IEEE Xplore](https://ieeexplore.ieee.org/document/8683041), [ResearchGate](https://www.researchgate.net/publication/331012502_Data_Augmentation_Strategies_for_Neural_Network_F0_Estimation)
