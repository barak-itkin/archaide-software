import itertools
import numpy as np

import tensorflow as tf


def tf_float_div(a, b, *args, **kwargs):
    return tf.div(
        tf.cast(a, dtype=tf.float32),
        tf.cast(b, dtype=tf.float32),
        *args, **kwargs)


def tf_if(cond, result, *args, **kwargs):
    return tf.cond(cond, result, tf.no_op(), *args, **kwargs)


def tf_assign_if(cond, ref, values):
    return tf_if(cond, tf.assign(ref, values))


class TFConfusion:
    def __init__(self, size, name, depth=None, dtype=tf.int32):
        """
        The matrix dimensions are:
        - First dimension - the number of the prediction.
        - Second dimension - the true label of the input sample.
        - Third dimension - the predicted label of the input sample.

        :param size:
        :param name:
        """
        self.size = size
        self.name = name
        self.depth = depth if depth is not None else self.size
        self.dtype = dtype

        with tf.name_scope(name):
            self.n = tf.Variable(
                0, dtype=self.dtype,
                trainable=False, name='n'
            )
            self.matrix = tf.Variable(
                tf.zeros((self.depth, self.size, self.size), dtype=self.dtype),
                trainable=False, name='matrix'
            )
            self.matrix_diag = tf.matrix_diag_part(
                self.matrix,
                name='matrix_diag'
            )
            self.input_histogram = tf.reduce_sum(
                self.matrix[0], axis=1,
                name='input_histogram'
            )
            self.output_histogram = tf.reduce_sum(
                self.matrix, axis=1,
                name='output_histogram'
            )
            self.false_positive_histogram = (
                self.output_histogram - self.matrix_diag
            )
            self.class_acc = tf_float_div(
                self.matrix_diag,
                tf.expand_dims(self.input_histogram, axis=0),
                name='class_acc'
            )
            self.cum_class_acc = tf.cumsum(
                self.class_acc, axis=0,
                name='cum_class_acc',
            )
            self.avg_acc = tf.reduce_mean(
                tf.boolean_mask(
                    self.class_acc,
                    tf.logical_not(tf.is_nan(self.class_acc))
                ),
                name='avg_acc'
            )
            self.acc = tf_float_div(
                tf.reduce_sum(self.matrix_diag),
                self.n,
                name='acc'
            )

    def make_is_not_empty_op(self, *args, **kwargs):
        with tf.name_scope(self.name):
            return tf.less(0, tf.reduce_sum(self.matrix), *args, **kwargs)

    def make_reset_op(self, *args, **kwargs):
        with tf.name_scope(self.name):
            _update_n = tf.assign(self.n, 0)
            with tf.control_dependencies([_update_n]):
                return tf.assign(
                    self.matrix, tf.zeros_like(self.matrix),
                    *args, **kwargs
                )

    def make_assign_op(self, data, *args, **kwargs):
        with tf.name_scope(self.name):
            return tf.assign(
                self.matrix,
                data,
                *args, **kwargs)

    def make_copy_op(self, other, *args, **kwargs):
        with tf.name_scope(self.name):
            return tf.assign(
                self.matrix,
                other.matrix,
                *args, **kwargs)

    def make_update_op(self, predicted_labels, labels):
        with tf.name_scope(self.name), tf.name_scope('update_op'):
            flat_labels = tf.reshape(labels, (-1, 1))
            flat_predictions = tf.reshape(predicted_labels, (-1, self.depth))
            n = tf.shape(flat_labels)[0]

            # We are going to update n X depth elements
            indices = tf.stack(
                (tf.tile(tf.reshape(tf.range(self.depth, dtype=self.dtype),
                                    (1, self.depth)),
                         (n, 1)),
                 tf.tile(flat_labels, (1, self.depth)),
                 flat_predictions
                 ),
                axis=-1
            )

            _update_n = tf.assign_add(self.n, tf.cast(n, dtype=self.dtype))

            with tf.control_dependencies([_update_n]):
                # Use tf.group to avoid any return value
                return tf.group(tf.scatter_nd_add(
                    self.matrix,
                    indices,
                    tf.ones((n, self.depth), dtype=self.dtype),
                    use_locking=True,
                    name='update_op'
                ))


class ConfusionMatrix:
    """3D confusion matrix - enabling recording multiple predictions per sample.

    The matrix dimensions are:
    - First dimension - the number of the prediction.
    - Second dimension - the true label of the input sample.
    - Third dimension - the predicted label of the input sample.
    """
    def __init__(self, size, depth=1):
        """Construct a 3D confusion matrix.

        Parameters
        ----------
        size: int
          The number of classes.
        depth: int
          The maximal number of predictions per sample.
        """
        self.matrix = np.zeros((depth, size, size))
        self.depth = depth
        self.size = size

    def record(self, labels, predictions):
        """Record prediction results.

        Parameters
        ----------
        labels: iterable of int
          The true labels of the samples.
        predictions: 2D iterable of int
          The first dimension corresponds to samples (same length as labels).
          The second dimension is for multiple predictions per sample.
        """
        n = len(labels)
        predictions = np.asarray(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape((n, -1))
        if n != predictions.shape[0] or self.depth != predictions.shape[1]:
            raise ValueError('Dimension mismatch!')
        # Pop another dimension into the predictions, if we have only one dimension
        for label, predictions in zip(labels, predictions):
            for i, prediction in enumerate(predictions):
                self.matrix[i, label, prediction] += 1

    def reset(self):
        self.matrix *= 0

    @property
    def input_histogram(self):
        return np.sum(self.matrix[0], 1)

    @property
    def prediction_histogram(self):
        return np.sum(self.matrix, 1)

    @property
    def n(self):
        return np.sum(self.matrix[0])

    @property
    def diagonal(self):
        return np.asarray([np.diag(layer) for layer in self.matrix])

    @property
    def hit(self):
        return np.sum(self.diagonal, 1)

    @property
    def miss(self):
        return self.n - self.hit

    @property
    def acc(self):
        return self.hit / self.n

    @property
    def class_acc(self):
        input = self.input_histogram
        nonz_input = input
        # Tile the 1D as rows of a matrix, with height by the depth
        deep_nonz_input = np.tile(
            nonz_input.reshape(1, self.size),
            [self.depth, 1]
        )
        return self.diagonal / deep_nonz_input

    @property
    def cumulative_class_acc(self):
        return np.cumsum(self.class_acc, 0)

    @property
    def cumulative_acc(self):
        return np.cumsum(self.acc)

    @property
    def cumulative_balanced_acc(self):
        return np.mean(self.cumulative_class_acc, 1)

    def __str__(self):
        return str(self.matrix)

    # Plot a confusion matrix, using a modified version of the code from
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plot(self, classnames=None, normalize=False, cmap=None, layer=0):
        import matplotlib.pyplot as plt
        cmap = cmap or plt.cm.Blues

        matrix = self.matrix[layer]
        if normalize:
            cm = matrix / matrix.sum(axis=1)[:, np.newaxis]
        else:
            cm = matrix.astype(np.int32)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        if classnames:
            tick_marks = np.arange(len(classnames))
            plt.xticks(tick_marks, classnames, rotation=45)
            plt.yticks(tick_marks, classnames)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
