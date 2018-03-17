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

    def predict_top_k_from_features(self, features, k, with_scores=False):
        # Get the scores per class for all the features
        # -> (sample_num) X (score of class)
        scores = self.classify_features_to_all(features)
        n_samples, n_classes = scores.shape

        # On dimension 1 (second dimension - samples), partition the scores
        # in descending order, and take the top k (first k)
        # (sample_num) X (class indices of top K scores)
        top_col_indices = np.argpartition(-scores, k, 1)[:, :k]

        # In `top_indices`, each row now contains column indices within that
        # row. To use these to index into scores, we just put these where we'd
        # usually write a column index.
        # However, we also need a matching array of row indices into the
        # scores in order to index this way.
        row_indices = np.tile(
            np.arange(n_samples).reshape(n_samples, 1),
            (1, k)
        )

        # (sample_num) X (scores of top K classes)
        top_scores = scores[row_indices, top_col_indices]

        # We now want to sort these top scores in descending order
        ordered_top_indices = np.argsort(-top_scores, 1)

        # And this is the ordering among the top classes (indices) per row
        result = top_col_indices[row_indices, ordered_top_indices]

        if not with_scores:
            return result
        else:
            return result, top_scores[np.arange(n_samples), ordered_top_indices]

    def predict_top_k(self, inputs, k, with_scores=False):
        return self.predict_top_k_from_features(
            self.compute_features(inputs), k, with_scores=with_scores)

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
