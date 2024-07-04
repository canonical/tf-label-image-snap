# TensorFlow Lite label image snap

This example bundles an example TensorFlow Lite script with all the required dependencies.
The base of this snap is core24, which is based on the current latest LTS release of Ubuntu, namely Ubuntu 24.04 Noble Numbat, and should provide the longest support.
The script is based on the [TensorFlow Lite Python image classification demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/python).

Core24 ships with Python 3.12, but TensorFlow Lite does not currently work with Python 3.12 due to dependencies that are [not yet updated](https://github.com/tensorflow/tensorflow/issues/62003).
The most reliable combination of dependencies that work on AMD64 and ARM64 are based on Python 3.8.
This snap includes Python 3.8 from the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).

## How to use

If the app is run without any arguments, it will use the included `mobilenet v1-1.0-224` model and classify an image of [Grace Hopper](https://en.wikipedia.org/wiki/Grace_Hopper) that is included in the snap.

```
$ tf-label-image
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
0.919721: 653:military uniform
0.017762: 907:Windsor tie
0.007507: 668:mortarboard
0.005419: 466:bulletproof vest
0.003828: 458:bow tie, bow-tie, bowtie
time: 28.502ms
```

Any other image can be specified using the `-i` argument. The snap has read access to the user's home directory, so any path under $HOME (or `~`) can be specified. Example:

```
$ tf-label-image -i ~/Downloads/parrot.jpeg
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
0.999824: 89:macaw
0.000119: 91:lorikeet
0.000013: 603:horizontal bar, high bar
0.000012: 703:parallel bars, bars
0.000010: 97:toucan
time: 25.774ms
```

## Advanced usage

This app provides command line arguments to override its default behaviour.

```
$ tf-label-image --help
usage: label_image_lite.py [-h] [-i IMAGE] [-m MODEL_FILE] [-l LABEL_FILE]
                           [--input_mean INPUT_MEAN] [--input_std INPUT_STD]
                           [--num_threads NUM_THREADS] [-e EXT_DELEGATE]
                           [-o EXT_DELEGATE_OPTIONS]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        image to be classified
  -m MODEL_FILE, --model_file MODEL_FILE
                        .tflite model to be executed
  -l LABEL_FILE, --label_file LABEL_FILE
                        name of file containing labels
  --input_mean INPUT_MEAN
                        input_mean
  --input_std INPUT_STD
                        input standard deviation
  --num_threads NUM_THREADS
                        number of threads
  -e EXT_DELEGATE, --ext_delegate EXT_DELEGATE
                        external_delegate_library path
  -o EXT_DELEGATE_OPTIONS, --ext_delegate_options EXT_DELEGATE_OPTIONS
                        external delegate options, format: "option1: value1;
                        option2: value2"
```

## Debugging

Python 3.8 and the bundled dependencies are exposed by the snap under the `tf-label-image.python` app.
This can be used to run custom Python scripts, or get an interactive Python shell.

```
$ tf-label-image.python
Python 3.8.19 (default, Apr 27 2024, 21:20:48)
[GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tflite_runtime
>>> tflite_runtime.__version__
'2.14.0'
>>>
```
