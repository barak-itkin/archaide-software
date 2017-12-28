import logging
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn import svm


import c3d.classification


def load_resnet_mean_bgr():
    raise NotImplementedError()


def resnet_preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    mean_bgr = load_resnet_mean_bgr()
    out = np.copy(img) * 255.0
    # swap channel from RGB to BGR
    if img.ndim == 3 and img.shape[-1] == 3:
        out = out[:, :, [2, 1, 0]]
    elif img.ndim == 4 and img.shape[-1] == 3:
        out = out[:, :, :, [2, 1, 0]]
    else:
        raise ValueError('Unknown image/batch dimension ' + str(img.shape))
    out -= mean_bgr
    return out


class Classifier(c3d.classification.FeatureClassifier):
    def __init__(self, dataset, resnet_dir, tf_session=None, cache_dir=None, max_train_samples=None, skip_validations=False):
        super(Classifier, self).__init__(dataset, tf_session)
        if not skip_validations and not os.path.isdir(resnet_dir):
            raise ValueError('Invalid resnet dir %s' % resnet_dir)
        self.resnet_dir = resnet_dir
        if not skip_validations and not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, 'decor_feature_cache.pickle')
        self.temp_cache_path = os.path.join(cache_dir, 'decor_feature_cache.temp.pickle')

        self.resnet_features_out = None
        self.resnet_images_in = None

        self.clf = None
        self.data_features = []
        self.data_ids = []

        self.feature_mean = None
        self.feature_std = None

        self.max_train_samples = max_train_samples

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

    def _compute_features(self, images):
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

    def _prepare(self):
        super(Classifier, self)._prepare()
        self.load_resnet()

    def _load_cache(self, temp=False):
        path = self.temp_cache_path if temp else self.cache_path
        if path and os.path.exists(path):
            logging.info('Loading cached image feautres')
            with open(path, 'rb') as f:
                self.data_features, self.data_ids, self.label_to_index = (
                        pickle.load(f)
                )
                return True
        return False

    def _save_cache(self, temp=False):
        path = self.temp_cache_path if temp else self.cache_path
        if not path:
            return

        with open(path if temp else path, 'wb') as f:
            pickle.dump(
                    (self.data_features, self.data_ids, self.label_to_index), f)

    def _has_enough_train_samples(self):
        if self.max_train_samples is None or self.max_train_samples <= 0:
            return False
        else:
            return len(self.data_features) >= self.max_train_samples

    def cache_features(self):
        # Did we already cache and finish the following computations?
        if self.feature_mean is not None:
            return

        # Only compute if we can't load from cache?
        if not self._load_cache():
            logging.info('Starting to cache image feautres')
            n_images = self.dataset.count()

            if self._load_cache(temp=True):
                existing = set(self.data_ids)
                logging.info('Loaded a temporary cache with %d images' % len(existing))
            else:
                existing = set()
            missing = set(self.dataset.file_ids_iter()) - existing

            if not self._has_enough_train_samples():
                b = 0
                for batch_ids, batchs_imgs in self.dataset.files_batch_iter(
                        batch_size=100, num_epochs=1, file_ids=missing):
                    self.data_ids.extend(batch_ids)
                    self.data_features.extend(self.compute_features(batchs_imgs))
                    logging.info('Processed %06d/%06d images', len(self.data_ids), n_images)
                    b += 1
                    if b % 10 == 0:
                        self._save_cache(temp=True)
                    if self._has_enough_train_samples():
                        break

                self._save_cache()

            logging.info('Done caching image feautres')

        # Regardless of how we got the cache, these stats should be computed.
        self.data_features = np.asarray(self.data_features)
        self.feature_mean = np.mean(self.data_features, 0)
        self.feature_std = np.std(self.data_features, 0)

    def prepare_features_for_clf(self, features):
        nonzero_std = self.feature_std + (self.feature_std == 0)
        return (np.asarray(features) - self.feature_mean) / nonzero_std

    def _classify_features_to_all(self, features):
        messy_scores = self.clf.decision_function(
            self.prepare_features_for_clf(features))
        # WARNING: THE SCORES RETURNED ARE IN THE ORDER CORRESPONDING TO
        #     self.clf.classes_
        # THIS MAY BE DIFFERENT THAN THE NATURAL ORDERING OF THE LABELS (I.E.
        # THE SCORES MAY BE FOR CLASSES IN A NON-SORTED ORDER).
        n_samples = len(features)
        result = -np.inf * np.ones((n_samples, self.n_classes))
        result[:, self.clf.classes_] = messy_scores
        return result

    def train_indices(self):
        return [i for i, id in enumerate(self.data_ids)
                if self.dataset.is_train_file(id)]

    def _train(self):
        self.cache_features()
        train_indices = self.train_indices()
        # Make sure no order exists in the input to SVM (in case the internal
        # implementation doesn't shuffle)
        np.random.shuffle(train_indices)
        # Use less train indices if specified (random because of shuffling)
        if self.max_train_samples is not None and self.max_train_samples > 0 and len(train_indices) > self.max_train_samples:
            train_indices = train_indices[:self.max_train_samples]

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
        test_indices = self.test_indices()

        X = self.prepare_features_for_clf(self.data_features[test_indices])
        Y = self.data_labels_numeric[test_indices]
        Y_predicted = self.clf.predict(X)
        return Y, Y_predicted

    # Implement pickle support
    def __getstate__(self):
        state = super(Classifier, self).__getstate__()
        state['cache_dir'] = self.cache_dir
        state['resnet_dir'] = self.resnet_dir
        state['clf'] = self.clf
        state['feature_mean'] = self.feature_mean
        state['feature_std'] = self.feature_std
        return state

    def __setstate__(self, state):
        self.__init__(
            state['dataset'], state['resnet_dir'],
            cache_dir=state.get('cache_dir', None),
            skip_validations=True
        )
        self.label_to_index = state['label_to_index']
        self.clf = state['clf']
        self.feature_mean = state['feature_mean']
        self.feature_std = state['feature_std']
