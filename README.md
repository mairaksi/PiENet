# PiENet
Pitch estimation network (PiENet) for noise-robust neural F0 estimation of speech signals

## Licence

## Dependencies
Tensorflow (tested with version 1.10)

Numpy

Scipy

## How to use pre-trained model:
### Method 1:
`python generate.py` : Perform pitch estimation to the `.wav` files specified in the text file `test_wavs.scp` and outputs the `.f0` files to the default folder located at `PRJDIR/f0/`.

### Method 2:
`python generate.py input_file.wav` : Perform pitch estimation to `input_file.wav`, output to `PRJDIR/f0/`.

`python generate.py input_list.scp` : Perform pitch estimation to all files specified in the text file `input_list.scp`, output to `PRJDIR/f0/`.

`python generate.py input_dir` : Perform pitch estimation to all `.wav` files found in the folder `input_dir`, output to `PRJDIR/f0/`.

### Method 3:
`python generate.py [input_file/input_list/input_dir] target_dir` : Same options as Method 2, output to `target_dir`.

## Reference:
[1] M. Airaksinen, J. Juvela, P. Alku and O. Räsänen: "Data augmentation strategies for neural F0 estimation", Proc. ICASSP 2019


