# archaide-software
Software deliverables for the ArchAIDE project

## Installation
The setup instructions were tested on Linux (Ubuntu 16.04) 64-bits. While they
should work with minor modifications also on Windows/Mac, this was not tested.

### Python
You will need to use Python >= 3.5.2

### Required Python packages
To run the code, you will need to have at least the following packages installed
in your environment:

* `numpy` >= 1.15.1
* `tensorflow-gpu` >= 1.10.0
    * It is possible also to use the CPU version (`tensorflow`), but it will be
      *much* slower, and so it's discouraged.
    * See notes about troubleshooting TensorFlow installation below.
* `scikit-image` (aka `skimage`) >= 0.13.1
* `scikit-learn` (aka `sklearn`) >= 0.19.1
* `Pillow` (aka `PIL`) >= 5.0.0

We recommend using `virtualenv` (or `conda`) to install these packages
in an isolated environment. Once inside your Python environment, all packages
can simply be installed with `pip`:

```
# Optional: create and use a virtualenv
virtualenv venv/ -p python3
source venv/bin/activate

# And now install the packages
pip install numpy matplotlib scikit-learn scikit-image tensorflow-gpu Pillow
```

If you haven't installed TensorFlow with GPU support previously, you may
encounter issues with the installation of TensorFlow. If indeed it does not
install smoothly, please follow the
[official instructions](https://www.tensorflow.org/install/).

Regardless of whether installation worked "smoothly" or not, **make sure GPU
support works properly** (if you installed TensorFlow with GPU support) before
proceeding to the next steps. You can do this by running the following command
and making sure it finishes without an error.

```
python -c 'import tensorflow'
```

If you are getting errors, see the
[common installation problems](https://www.tensorflow.org/install/install_linux#common_installation_problems)
of the guide.

### ResNet dependency
In order to run the appearance classifier, we rely on a pre-trained
version of the ResNet-101 network. This can be obtained for TensorFlow
by following the instructions on the
[tensorflow-resnet](https://github.com/ry/tensorflow-resnet) project. We
recommend using the already converted model they supply in the torrent
link (and extracting the `tensorflow-resnet-pretrained-20160509` folder
here inside the root folder of the repository). If the torrent is not
available for some reason, follow the conversion instructions on their
documentation.

## Usage

### Appearance based similarity

#### Foreground extraction

Due to the reasons detailed in D6.3, to achieve improved accuracy and reduce the
bias towards specific backgrounds for specific classes, we use a foreground
extraction process - a process that automatically extracts sherds from their
backgrounds, before using them for training the classification.

To do this, we first use the following tool on the folder containing the original
data:

```
python -m c3d.appearance2.fg_extract \
  path/to/raw/data/folder \
  --alternate_base=path/to/extracted/data/folder \
  --num_workers=10
```

This will create a new folder with the foreground extracted version of the
images. This is the folder you should use for the actual training.

Note the parameter `num_workers` - use this to manage the number of workers
that run in parallel to process the image. Increasing this can cause the
processing to run faster, but it requires more CPUs from your computer.
We recommend setting this to not more than
`<The number of CPUs on your computer> - 2`.

#### Training

Train the model on images in the folder (leaving out some images for testing).
The label of each image will be the name of the folder containing it:

```sh
# Create a directory for saving intermediate training results (used to continue
# training if it was interrupted)
mkdir path/to/cache

# Train the model
python -m c3d.appearance2 \
  path/to/model.pkl path/to/extracted/data/folder \
  --resnet-dir=path/to/resnet-dir \
  train
  --cache_dir path/to/cache
```

#### Classifying

Classify all images in a given folder. For each image, list the top 3 guesses:

```
python -m c3d.appearance2 \
  path/to/model.pkl path/to/extracted/data/folder \
  --k=3
  classify
```

#### Testing

Evaluate our training - test the model on the images that were left out for
testing, listing top 5 guesses:

```
python -m c3d.appearance2 \
  path/to/model.pkl path/to/extracted/data/folder \
  --k=5
  test
```

Note: The test should be carried out on the same folder with the source images.

#### Further help

Additional arguments can be found using the `--help` flag:

```
python -m c3d.appearance2 --help
```

### Shape based similarity

Train the model on all images in the folder. The label of each image will be
the name of the folder containing it:
```
python -m c3d.shape example.model path/to/data train
```
The data can be in one of two flavors:

1. JSON files representing c3d.datamodel.Fracture objects. In that case, the
   file names must end with `.fracture.json`
1. Single-channel 256x256 grayscale images, with white representing the
   fracture and black representing the background.

Note that **it is OK to interrupt the training with Ctrl+C** - in the case the
model that'll be saved is what was trained so far. Furthermore, running the
same command again will continue training where it has previously stopped.

Classify all images in a given folder. For each image, list the top 3 guesses:
```
python -m c3d.shape example.model path/to/data --k 3 classify
```

Evaluate our training - check the accuracy of our model on the data that was
left out for testing during the training process:
```
python -m c3d.shape example.model path/to/data --k 3 classify
```

Additional arguments can be found using the `--help` flag:
```
python -m c3d.shape --help
```
