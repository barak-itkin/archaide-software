# archaide-software
Software deliverables for the ArchAIDE project

## Installation
The setup instructions were tested on Linux (Ubuntu 16.04) 64-bits. While they
should work with minor modifications also on Windows/Mac, this was not tested.

### Python
You will need to use Python >= 3.5.2

### IMPORTANT - Compatiability note (2021-11-27)

*   As of 2021-11-27, the project does not work with TensorFlow v2 or Python
    versions newer than 3.6.X.
*   This is because of our dependency on
    [PointNet](https://github.com/charlesq34/pointnet) (in our specific fork,
    [barak-itkin/pointnet-archaide](https://github.com/barak-itkin/pointnet-archaide)
    which was not migrated to TensorFlow v2.
*   I would be happy to help on guiding others to migrate the code (for both
    PointNet and ArchAIDE), but unfortunately I don't have the resources to do
    so myself; I graduated from university multiple years ago, and so I don't
    have access to the needed computing resources, and I'm also limited in the
    time I have to continue working on this code.

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

### PointNet dependency
The current architecture of the shape-based classification, is based on
[PointNet](https://github.com/charlesq34/pointnet). Our code uses a
modified version of PointNet, which was modified for our own needs, and
is available at [PointNet-ArchAIDE](https://github.com/barak-itkin/pointnet-archaide).
All we need for now, is just to clone this repository and then set it up
to use revision **@`48ce752`** by doing:

```
# Clone the repository and then enter it's directory
git checkout 48ce752
```

## Usage

### Appearance based similarity

#### Foreground extraction

Due to the reasons detailed in D6.3, to achieve improved accuracy and reduce the
bias towards specific backgrounds for specific classes, we use a foreground
extraction process - a process that automatically extracts sherds from their
backgrounds, before using them for training the classification.

To do this, we first use the following tool on the folder containing the original
data:

```sh
python -m c3d.appearance2.fg_extract \
  path/to/raw/data/folder \
  --alternate_base path/to/extracted/data/folder \
  --num_workers 10
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
  --resnet-dir path/to/resnet-dir \
  train
  --cache_dir path/to/cache
```

#### Classifying

Classify all images in a given folder. For each image, list the top 3 guesses:

```sh
python -m c3d.appearance2 \
  path/to/model.pkl path/to/extracted/data/folder \
  --k 3
  classify
```

#### Testing

Evaluate our training - test the model on the images that were left out for
testing, listing top 5 guesses:

```sh
python -m c3d.appearance2 \
  path/to/model.pkl path/to/extracted/data/folder \
  --k 5
  test
```

Note: The test should be carried out on the same folder with the source images.

#### Further help

Additional arguments can be found using the `--help` flag:

```sh
python -m c3d.appearance2 --help
```

### Shape based similarity

#### Generating the synthetic sherds

To train the shape recognition model, we first need to generate the training
data - lots of synthetic sherd fractures. To do this, we use the following tool
on the folder containing the profiles to train on:

```
python -m c3d.pipeline.fracture2 \
  path/to/profile-svgs/data/folder \
  path/to/sherds/data/folder \
  2000
```

This will create a new folder with the specified number of synthetic sherds
generated for each profile drawing (SVG file). This is the data we'll use in the
training process.

#### Training

To train the model on the synthetic sherds in the folder, run

```
# Create a directory for saving intermediate training results (used to continue
# training if it was interrupted)
mkdir path/to/cache

# Train the model
PYTHONPATH=path/to/pointnet-archaide:$PYTHONPATH python -m c3d.shape2 \
  path/to/model.pkl path/to/sherds/data/folder \
  [ --label-mapping-file path/to/mapping/file.json \ ]
  train \
  --cache_dir path/to/cache \
  [ --eval_set path/to/sherd-svgs/data/folder ]
```

Note that we need to add the path to the `pointnet-archaide` repository to the
`PYTHONPATH` environment variable. Furthermore, if we want to specify that
multiple profiles are equivalent (for example, to specify that all subtypes of a
specific top-level type are equivalent, for training a top-level classifier), we
need to specify a path to a label mapping file. The label mapping is a JSON file
with the following structure:

```json
{
  "profile01_name": "new_name01",
  "profile02_name": "new_name02",
  ...
}
```

For convinience, we include an example such file mapping all subtypes of
Terra Sigillata (from the Conspectus) to their top level profiles. This file
is `terra-sigillata-complete-top-level.json` which is provided in this
repository. Note that we excluded subtypes that only show a partial shape
(and not a full profile); **by excluding profiles from the label mapping file,
we remove them from the training process.**

Another argument we specified is an evaluation set - if specified, this should
be the path to a folder containing SVGs extracted from real sherds. The SVGs
should be organized under folders with the same names as in the training set,
and they will be periodically used to evaluate the classifier accuracy on them,
providing a measure of the accuracy on real data.

#### Classifying

To classify all sherd SVGs in a given folder, listing the top 3 guesses for
each input, run the following command:

```sh
PYTHONPATH=path/to/pointnet-archaide:$PYTHONPATH python -m c3d.shape2 \
  path/to/model.pkl path/to/sherds/data/folder \
  [ --label-mapping-file path/to/mapping/file.json \ ]
  --svg-inputs \
  --k 3 \
  classify
```

This command will also print the accuracy of the classification (assuming the
folder names match the names used during the training).
