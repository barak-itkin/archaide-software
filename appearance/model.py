import os
import logging
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
import pickle
from sklearn import svm

from .dataset import ImageDataset


# returns image of shape [224, 224, 3]
# [height, width, depth]
def box_crop(img, size=224):
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img


def symmetric_pad(start, end, width):
    size = end - start
    if size > width:
        raise NotImplementedError('Only supports padding, not cropping')
    elif size == width:
        return np.arange(start, end)
    else:
        pad = width - size
        pad_s = pad // 2
        pad_e = pad - pad_s
        result = np.zeros(width, dtype=np.int32) + start
        result[-pad_e:] = end - 1
        result[pad_s:(pad_s+size)] = np.arange(start, end)
        return result


def box_fit(img, size=224):
    if img.shape[0] > img.shape[1]:
        pad_img = img[:, symmetric_pad(0, img.shape[1], img.shape[0]), :]
    else:
        pad_img = img[symmetric_pad(0, img.shape[0], img.shape[1]), :, :]
    resized_img = skimage.transform.resize(pad_img, (size, size))
    return resized_img


def load_resnet_mean_bgr():
    raise NotImplementedError()


def resnet_preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    mean_bgr = load_resnet_mean_bgr()
    out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out -= mean_bgr
    return out


def multiply(function, iterable):
    for val in iterable:
        for result in function(val):
            yield result


def img_scale(img):
    yield box_fit(img)
    yield box_fit(img, 284)[20:244,20:244,:]
    yield box_fit(img, 324)[40:264,40:264,:]
    yield box_crop(img)


def img_augment(img):
    yield img
    #yield img + 0.01 * np.random.rand(*img.shape)
    yield np.flip(img, 0)
    yield np.flip(img, 1)


def reverse_dict(src):
    return dict((val, key) for (key, val) in src.items())


class Classifier:
    def __init__(self, dataset, resnet_dir, tf_session=None, cache_path=None):
        self._tf_session = tf_session
        self.dataset = dataset
        self.resnet_dir = resnet_dir
        self.cache_path = cache_path

        self.resnet_features_out = None
        self.resnet_images_in = None

        self.clf = None
        self.data_features = []
        self.data_ids = []

        self.label_to_index = {}
        self.index_to_label = {}

        self.feature_mean = None
        self.feature_std = None

    @property
    def tf_session(self):
        if not self._tf_session:
            self._tf_session = self._tf_session or tf.Session()
        return self._tf_session

    @property
    def data_labels(self):
        return [id.label for id in self.data_ids]

    @property
    def data_labels_numeric(self):
        return np.asarray([self.label_to_index[id.label]
                          for id in self.data_ids])

    def resnet_model_file_path(self, n_layers, suffix):
        n_layers = str(n_layers)  # In case this was passed as an integer
        assert n_layers.isdigit()  # Avoid silly mistakes and path issues
        name = 'ResNet-L{n_layers}{suffix}'.format(
                n_layers=n_layers, suffix=suffix
        )
        return os.path.join(self.resnet_dir, name)

    def compute_resnet_features(self, images):
        if self.resnet_features_out is None:
            self.load_resnet()
        return self.tf_session.run(self.resnet_features_out, {
            self.resnet_images_in: images
        })

    def load_resnet(self):
        if self.resnet_features_out is not None:
            return

        saver = tf.train.import_meta_graph(
                self.resnet_model_file_path(101, '.meta'))
        saver.restore(
                self.tf_session, self.resnet_model_file_path(101, '.ckpt'))

        def get_output(name):
            return self.tf_session.graph.get_operation_by_name(name).outputs[0]

        # Take the latest features from each block, and average them on the
        # width and height of the image, to get a single feature vector from
        # each block per image in the batch
        f2im = get_output('scale2/block3/Relu')
        f2 = tf.reduce_mean(f2im, reduction_indices=[1, 2], name='avg_pool2')

        f3im = get_output('scale3/block4/Relu')
        f3 = tf.reduce_mean(f3im, reduction_indices=[1, 2], name='avg_pool3')

        f4im = get_output('scale4/block23/Relu')
        f4 = tf.reduce_mean(f4im, reduction_indices=[1, 2], name='avg_pool4')

        f5im = get_output('scale5/block3/Relu')
        f5 = tf.reduce_mean(f5im, reduction_indices=[1, 2], name='avg_pool5')

        # Now concatenate all the features from all the different blocks, along
        # dimension 1 (where 0 is the index in the batch).
        self.resnet_features_out = tf.concat([f2, f3, f4, f5], 1, 'concat_feat')

        # The input placeholder
        self.resnet_images_in = get_output('images')

    def augment_image(self, img):
        imgs = multiply(img_scale, [img])
        imgs = multiply(img_augment, imgs)
        return imgs

    def _load_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            logging.info('Loading cached image feautres')
            with open(self.cache_path, 'rb') as f:
                self.data_features, self.data_ids, self.label_to_index = (
                        pickle.load(f)
                )
                return True
        return False

    def _save_cache(self):
        if not self.cache_path:
            return

        with open(self.cache_path, 'wb') as f:
            pickle.dump(
                    (self.data_features, self.data_ids, self.label_to_index), f)

    def cache_features(self):
        # Did we already cache and finish the following computations?
        if self.feature_mean is not None:
            return

        # Only compute if we can't load from cache?
        if not self._load_cache():
            logging.info('Starting to cache image feautres')
            processed = 0
            n_images = self.dataset.count()

            for batch_ids, batchs_imgs in self.dataset.files_batch_iter(
                    batch_size=10, num_epochs=1):
                # Create a batch of images to evaluate in ResNet each time,
                # instead of evaluating them one by one, to allow higher
                # efficiency.
                resnet_input_images = []
                for file_id, img in zip(batch_ids, batchs_imgs):
                    img_augments = list(self.augment_image(img))
                    resnet_input_images.extend(img_augments)
                    self.data_ids.extend([file_id] * len(img_augments))
                    processed += 1
                resnet_input_images = np.asarray(resnet_input_images)
                self.data_features.extend(
                        self.compute_resnet_features(resnet_input_images))
                logging.info('Processed %06d/%06d images', processed, n_images)

            logging.info('Done caching image feautres')
            all_labels = set(self.data_labels)
            self.label_to_index = dict(
                (label, i) for i, label in enumerate(sorted(all_labels))
            )
            self.index_to_label = reverse_dict(self.label_to_index)

            self._save_cache()

        # Regardless of how we got the cache, these stats should be computed.
        self.data_features = np.asarray(self.data_features)
        self.feature_mean = np.mean(self.data_features, 0)
        self.feature_std = np.std(self.data_features, 0)

    def prepare_features_for_clf(self, features):
        return (np.asarray(features) - self.feature_mean) / self.feature_std

    def predict_top_k_from_features(self, resnet_features, k):
        distances = self.clf.decision_function(
                self.prepare_features_for_clf(resnet_features))
        n_classes = distances.shape[1]
        top_indexes = np.argpartition(distances, n_classes - k, 1)[:, -k:]
        # Set a minus on the distances to make the sort descending
        ordered_top_indexes = np.argsort(-distances.take(top_indexes))
        return top_indexes.take(ordered_top_indexes)

    def predict_top_k(self, images, k):
        return self.predict_top_k_from_features(
                self.compute_resnet_features(images), k)

    def train_indices(self):
        return [i for i, id in enumerate(self.data_ids)
                if self.dataset.is_train_file(id)]

    def train(self):
        self.cache_features()
        train_indices = self.train_indices()

        X = self.prepare_features_for_clf(self.data_features[train_indices])
        Y = self.data_labels_numeric[train_indices]

        self.clf = svm.LinearSVC(C=0.001, multi_class='ovr')
        logging.info('Starting to train SVM classifier')
        self.clf.fit(X, Y)
        logging.info('Finished training SVM classifier')

        Y_predicted = self.clf.predict(X)
        return Y, Y_predicted

    def test_indices(self):
        return [i for i, id in enumerate(self.data_ids)
                if not self.dataset.is_train_file(id)]

    def test(self):
        self.cache_features()
        test_indices = self.train_indices()

        X = self.prepare_features_for_clf(self.data_features[test_indices])
        Y = self.data_labels_numeric[test_indices]
        Y_predicted = self.clf.predict(X)
        return Y, Y_predicted

    # Implement pickle support
    def __getstate__(self):
        return {
            'dataset': self.dataset,
            'cache_path': self.cache_path,
            'resnet_dir': self.resnet_dir,
            'clf': self.clf,
            'label_to_index': self.label_to_index,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std
        }
        return state

    def __setstate__(self, state):
        self.__init__(
            state['dataset'], state['resnet_dir'],
            cache_path=state.get('cache_path', None)
        )
        self.clf = state['clf']
        self.label_to_index = state['label_to_index']
        self.index_to_label = reverse_dict(self.label_to_index)
        self.feature_mean = state['feature_mean']
        self.feature_std = state['feature_std']

