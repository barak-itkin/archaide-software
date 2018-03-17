import numpy as np


def sigmoid(X, theta=1.0):
    return np.exp(X) / (1 + np.exp(X))


# A softmax implementation in Numpy, based on the code from:
#   https://nolanbconaway.github.io/blog/2017/softmax-numpy
# The code was modified to operate on the last axis by default, to have
# consistent behavior with Tensorflow.
def softmax(X, theta=1.0, axis=-1):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0.
    axis (optional): axis to compute values along. Default is the
        last axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = -1

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    print('Softmax: %s -> %s' % (X.shape, y.shape))
    return p
