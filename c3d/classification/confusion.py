import itertools
import numpy as np


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
        nonz_input = input + (input == 0)
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
