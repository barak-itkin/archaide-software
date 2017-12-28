import collections
import numpy
import os
import skimage.io


AUGMENT_PREFIX = '$augment.'
AUGMENT_SUFFIX = '$$'


FileId = collections.namedtuple('FileId', 'label id augment')
NamedSample = collections.namedtuple('NamedSample', 'name sample')


class Dataset:
    def __init__(self, data_root, train_to_test_ratio=None):
        self.train_to_test_ratio = train_to_test_ratio
        self.has_test = train_to_test_ratio is not None
        self.data_root = os.path.abspath(data_root)
        self.shuffle_seed = 0
        self.num_sherds = 1
        self.sherd_id = 0

    def shuffle(self):
        self.shuffle_seed += 1

    def label_from_path(self, dirpath, filename):
        # By default, the label of a file is its dirname (in the dataset)
        return os.path.relpath(dirpath, self.data_root)

    def id_from_path(self, dirpath, filename):
        # By default, the ID of a file is its name
        return filename

    # And following that logic, we can reconstruct the path
    def file_name(self, file_id):
        # Override according to id_from_path / label_from_path
        return file_id.id

    def file_dir(self, file_id):
        # Override according to id_from_path / label_from_path
        return file_id.label

    def file_path(self, file_id):
        dirname = self.file_dir(file_id)
        fname = self.file_name(file_id)

        if file_id.augment:
            augment_part = AUGMENT_PREFIX + file_id.augment + AUGMENT_SUFFIX
            if '.' in fname:
                augment_part += '.' + fname.split('.', 1)[-1]
        else:
            augment_part = ''

        return os.path.join(
            self.data_root, dirname, fname + augment_part
        )

    def file_filter(self, dirname, filename):
        return True

    def file_id_filter(self, file_id):
        return True

    def file_ids_iter(self):
        for root, dirs, files in os.walk(self.data_root):
            dirpath = os.path.join(self.data_root, root)
            # Make the dirpath canonical so names can be extracted reliably
            dirpath.rstrip(os.path.sep)
            for filename in files:
                if AUGMENT_PREFIX in filename:
                    filename, augment = filename.split(AUGMENT_PREFIX, 1)
                    assert AUGMENT_SUFFIX in augment
                    augment, ext = augment.rsplit(AUGMENT_SUFFIX, 1)
                    if ext:
                        assert ext[0] == '.'
                        assert filename.endswith(ext)
                else:
                    augment = None

                if not self.file_filter(dirpath, filename):
                    continue
                f = FileId(
                    label=self.label_from_path(dirpath, filename),
                    id=self.id_from_path(dirpath, filename),
                    augment=augment
                )
                if self.file_id_filter(f) and hash(f) % self.num_sherds == self.sherd_id:
                    yield f

    def file_ids_train_iter(self):
        for f in self.file_ids_iter():
            if self.is_train_file(f):
                yield f

    def file_ids_test_iter(self):
        for f in self.file_ids_iter():
            if self.is_train_file(f):
                yield f

    def count(self):
        return sum(1 for f_id in self.file_ids_iter())

    def prepare_file(self, file_id):
        raise NotImplementedError()

    def files_batch_iter(self, batch_size, num_epochs, file_ids=None,
                         no_prepare=False):
        if file_ids is None:
            file_ids = self.file_ids_iter()
        file_ids = list(file_ids)
        n = len(file_ids)
        for i in range(num_epochs):
            numpy.random.shuffle(file_ids)
            for b in range(0, n, batch_size):
                start = b
                end = min(n, b + batch_size)
                ids = file_ids[start:end]
                if no_prepare:
                    yield ids
                else:
                    yield ids, [
                        self.prepare_file(f_id) for f_id in ids
                    ]

    def files_iter(self, num_epochs=1, file_ids=None, no_prepare=False):
        for batch_ids, batch_files in self.files_batch_iter(1, num_epochs, file_ids, no_prepare):
            for id, f in zip(batch_ids, batch_files):
                yield id, f

    def is_train_file(self, file_id):
        return (not self.has_test
                or hash((file_id.unaugmented(), self.shuffle_seed)) % self.train_to_test_ratio
                    != 0)

    def is_test_file(self, file_id):
        return not self.is_train_file(file_id)

    def files_batch_train_iter(self, *args, **kwargs):
        return self.files_batch_iter(
                file_ids=self.file_ids_train_iter(), *args, **kwargs
        )

    def files_batch_test_iter(self, *args, **kwargs):
        return self.files_batch_iter(
                file_ids=self.file_ids_test_iter(), *args, **kwargs
        )

    def all_labels(self):
        return set(
            f_id.label for f_id in self.file_ids_iter()
        )


def augment(named_samples, augment_function):
    for ns in named_samples:
        for augmented in augment_function(ns.sample):
            new_name = (
                (ns.name or augmented.name) if not (augmented.name and ns.name)
                else '%s.%s' % (ns.name, augmented.name)
            )
            yield NamedSample(new_name, augmented.sample)


def make_augmented_id(file_id, augment_name):
    if not augment_name:
        return file_id
    else:
        return FileId(file_id.label, file_id.id, augment_name)


IMAGE_EXTENSIONS = set(['bmp', 'png', 'tiff', 'tif', 'jpg', 'jpeg'])


class ImageDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)
        self.filters = []

    def prepare_file(self, file_id):
        img = skimage.io.imread(self.file_path(file_id))
        for f in self.filters:
            img = f(img)
        return img

    def file_filter(self, dirpath, filename):
        return ('.' in filename
                and filename.split('.')[-1].lower() in IMAGE_EXTENSIONS)


def im2rgb(img):
    if len(img.shape) == 3:  # Row X Column X Channel
        if img.shape[-1] == 3:  # RGB
            return img
        elif img.shape[-1] == 4:  # RGBA
            return img[:, :, :3]
    elif len(img.shape) == 2:  # Row X Column
        return np.repeat(img[:, :, np.newaxis], 3, axis=2)
    raise ValueError('Unexpected image shape %s' % img.shape)
