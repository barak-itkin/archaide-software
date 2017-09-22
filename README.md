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

Train the model on all images in the folder. The label of each image will be
the name of the folder containing it:
```
python -m c3d.appearance train path/to/data --model example.model
```

Classify all images in a given folder. For each image, list the top 3 guesses:
```
python -m c3d.appearance classify path/to/image/folder --model example.model --k 3
```

Evaluate our training - split the data into train and test. Train on the train
data, and evaluate our accuracy on the test data. Check the accuracy on the
top 3 guesses, and repeat this train-eval process 5 times:
```
python -m c3d.appearance eval path/to/images --k 3 --num-runs 5
```

Additional arguments can be found using the `--help` flag:
```
python -m c3d.appearance --help
```
