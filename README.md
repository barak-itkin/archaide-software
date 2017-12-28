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

* `numpy` >= 1.13.0
* `tensorflow-gpu` >= 1.2.0
    * It is possible also to use the CPU version (`tensorflow`), but it will be
      *much* slower, and so it's discouraged.
    * See notes about troubleshooting TensorFlow installation below.
* `matplotlib` >= 2.0.2
* `scikit-image` (aka `skimage`) >= 0.13.0
* `scikit-learn` (aka `sklearn`) >= 0.19.0

We recommend using `virtualenv` (or `conda`) to install these packages
in an isolated environment. Once inside your Python environment, all packages
can simply be installed with `pip`:

```
# Optional: create and use a virtualenv
virtualenv venv/ -p python3
source venv/bin/activate

# And now install the packages
pip install numpy matplotlib scikit-learn scikit-image tensorflow-gpu
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
of the guide. Specically you may want to use TensorFlow 1.2.0 explicitly if you
get issues about failing to load libcudnn:

```
pip uninstall tensorflow-gpu
pip install tensorflow-gpu==1.2.0
```

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

#### Data Augmentation

To acheive improved accuracy with small amounts of images, we use an
*augmentation* process - that is, enriching the collection of images by creating
more versions (with different scales/croppings) of each original image.

To do this, we first use the following tool on the folder containing the original
data:

```
python -m c3d.appearance.make_dataset \
  path/to/raw/data/folder \
  path/to/augmented/data/folder
```

This will create the new augmented folder with many more versions of the images.
You should use this augmented folder for training.

#### Training

Train the model on images in the folder (leaving out some images for testing).
The label of each image will be the name of the folder containing it:

```
python -m c3d.appearance \
  example.model path/to/data \
  --resnet-dir=path/to/resnet-dir \
  train
```

#### Classifying

Classify all images in a given folder. For each image, list the top 3 guesses:

```
python -m c3d.appearance \
  example.model path/to/data \
  --k=3
  classify
```

Note: If the resnet dir is not were it was before and fails to load, you can
specify it again using `--resnet_dir` (before the `classify` action).

#### Testing

Evaluate our training - test the model on the images that were left out for
testing, listing top 5 guesses:

```
python -m c3d.appearance \
  example.model path/to/data \
  --k=3
  test
```

Note: The test should be carried out either on the folder with the source images
(OK even if you have an augmented folder, and this is preferred) or on the
augmented folder (discouraged).

#### Further help

Additional arguments can be found using the `--help` flag:

```
python -m c3d.appearance --help
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
