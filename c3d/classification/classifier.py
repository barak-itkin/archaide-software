import numpy as np
import tensorflow as tf

from c3d.util.doubledict import DoubleDict


class FeatureClassifier:
    def __init__(self, dataset, tf_session=None, label_to_index=None):
        self._tf_session = tf_session
        self.dataset = dataset
        self.label_to_index = DoubleDict(label_to_index or {})
        self.prepared = False

    @property
    def n_classes(self):
        self.cache_label_to_index()
        return len(self.label_to_index)

    @property
    def index_to_label(self):
        return self.label_to_index.reverse

    def record_label(self, label):
        if label in self.label_to_index:
            return
        index = len(self.label_to_index)
        self.label_to_index[label] = index
        self.index_to_label[index] = label

    def rename_label(self, old_label, new_label):
        if not old_label in self.label_to_index:
            raise ValueError('Unknown label %s' % old_label)
        if not new_label in self.label_to_index:
            raise ValueError('Label %s already exists' % new_label)

        index = self.label_to_index[old_label]
        del self.label_to_index[old_label]

        self.label_to_index[new_label] = index
        self.index_to_label[index] = new_label

    @property
    def tf_session(self):
        if not self._tf_session:
            self._tf_session = self._tf_session or tf.get_default_session() or tf.Session()
        return self._tf_session

    def predict_top_k_from_features(self, features, k):
        # (sample_num) X (score of class)
        scores = self.classify_features_to_all(features)
        n_classes = scores.shape[1]
        # (sample_num) X (classes of top K scores)
        top_indices = np.argpartition(scores, n_classes - k, 1)[:, -k:]
        # (sample_num) X (indices to sort the rows of `top_indices` rows, by
        #                 descending score)
        # Set a minus on the distances to make the sort descending
        ordered_top_indices = np.argsort(-scores.take(top_indices))
        # (sample_num) X (indices of top K scores, by descending score)
        return np.array([
            top_sample_indices.take(ordered_sample_indices)
            for top_sample_indices, ordered_sample_indices in zip(
                top_indices, ordered_top_indices
            )
        ])

    def predict_top_k(self, inputs, k):
        return self.predict_top_k_from_features(
                self.compute_features(inputs), k)

    def cache_label_to_index(self):
        if not self.label_to_index:
            for label in sorted(list(self.dataset.all_labels())):
                self.record_label(label)

    def _prepare(self):
        """Method to call before training/testing for the first time."""
        self.cache_label_to_index()

    def prepare(self):
        if self.prepared:
            return
        self._prepare()
        self.prepared = True

    def _train(self):
        raise NotImplementedError()

    def train(self):
        self.prepare()
        return self._train()

    def _compute_features(self, inputs):
        raise NotImplementedError()

    def compute_features(self, inputs):
        self.prepare()
        return self._compute_features(inputs)

    def _classify_features_to_all(self, features):
        raise NotImplementedError()

    def classify_features_to_all(self, features):
        self.prepare()
        return self._classify_features_to_all(features)

    # Implement pickle support
    def __getstate__(self):
        return {
            'dataset': self.dataset,
            'label_to_index': self.label_to_index,
        }

    def __setstate__(self, state):
        raise NotImplementedError()
