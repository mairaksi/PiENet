# PiENet
Pitch estimation network (PiENet) for noise-robust neural F0 estimation of speech signals. The best performing model from [1] ('GTE-AUG') supplied as a pre-trained model. The model was trained using additional and convolutional noise augmentation, as well as vocoder-based ground truth enhancement (see publication for further details).

Article found in:
[IEEE Xplore](https://ieeexplore.ieee.org/document/8683041),
[ResearchGate](https://www.researchgate.net/publication/331012502_Data_Augmentation_Strategies_for_Neural_Network_F0_Estimation)

## Licence
Distributed under the Apache 2.0 license. See LICENSE for further details.

## Dependencies
* Tensorflow (tested with version 1.10)
* Numpy
* Scipy

## How to use pre-trained model:
The pre-trained model is trained for speech sampled at 16 kHz, with 10 ms frame shift and a F0 range of 50 to 500 Hz.

### Method 1:
* `python generate.py` : Perform pitch estimation to the `.wav` files specified in the text file `test_wavs.scp` and outputs the `.f0` files to the default folder located at `./f0/`.

### Method 2:
* `python generate.py input_file.wav` : Perform pitch estimation to `input_file.wav`, output to `./f0/`.

* `python generate.py input_list.scp` : Perform pitch estimation to all files specified in the text file `input_list.scp`, output to `./f0/`.

* `python generate.py input_dir` : Perform pitch estimation to all `.wav` files found in the folder `input_dir`, output to `./f0/`.

### Method 3:
* `python generate.py [input_file/input_list/input_dir] target_dir` : Same options as Method 2, output to `target_dir`.

## How to train your own model:
* Supply list of train data files in `train_wavs.scp` for input data, and the corresponding target F0s in `train_f0s.scp` (datatype raw float32 for F0 files). Note that the number of frames in the F0 files must match the present method's framing convention (Nframes = ceil(wav_length/hop), signal zero-padded with winlen/2 samples from start and beginning).

* If you want to use additive noise augmentation with sampled noise from a database, supply the file paths in `noiselist.scp` and edit the code in `train.py` line 54 to `noise_samples = augmentation.load_noise_samples(noise_wav_scp=noiselist.scp)`.

* For vocoder-based augmentation techniques, you need to download and the vocoder of your choice. The GlottDNN vocoder used in [1] can be found in https://github.com/ljuvela/GlottDNN. For ground truth enhancement, process the train wavs offline with the vocoder. The code for diversity augmentation is not provided in the current release of the method.


## Reference: 
[1] M. Airaksinen, L. Juvela, P. Alku and O. Räsänen: "Data augmentation strategies for neural F0 estimation", Proc. ICASSP 2019

Available: [IEEE Xplore](https://ieeexplore.ieee.org/document/8683041),
[ResearchGate](https://www.researchgate.net/publication/331012502_Data_Augmentation_Strategies_for_Neural_Network_F0_Estimation)


