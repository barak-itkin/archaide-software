import collections
import numpy
import os
import skimage.io


FileId = collections.namedtuple('FileId', 'label id')


class Dataset:
    def __init__(self, data_root, train_to_test_ratio=None):
        self.train_to_test_ratio = train_to_test_ratio
        self.has_test = train_to_test_ratio is not None
        self.data_root = data_root
        self.shuffle_seed = 0

    def shuffle(self):
        self.shuffle_seed += 1

    def label_from_path(self, dirpath, filename):
        # By default, the label of a file is its dirname. We assume there is
        # no nesting of folders to deeper levels.
        return os.path.basename(dirpath)

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
        return os.path.join(
                self.data_root,
                self.file_dir(file_id), self.file_name(file_id)
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
                if not self.file_filter(dirpath, filename):
                    continue
                f = FileId(
                        self.label_from_path(dirpath, filename),
                        self.id_from_path(dirpath, filename)
                )
                if self.file_id_filter(f):
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

    def is_train_file(self, file_id):
        return (not self.has_test
                or hash((file_id, self.shuffle_seed)) % self.train_to_test_ratio
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


IMAGE_EXTENSIONS = set(['bmp', 'png', 'tiff', 'tif', 'jpg', 'jpeg'])


class ImageDataset(Dataset):
    def prepare_file(self, file_id):
        return skimage.io.imread(self.file_path(file_id))

    def file_filter(self, dirpath, filename):
        return ('.' in filename
                and filename.split('.')[-1].lower() in IMAGE_EXTENSIONS)
