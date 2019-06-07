# Partially based on https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import io
import numpy as np
import os
import pickle
import re
import tensorflow as tf
import time

from c3d.classification import ConfusionMatrix


NAME_PATTERN = re.compile(r'^[\w\-_\.]+$')


def max_enumerate(values, limit):
    return zip(range(limit), values)


def tensor_summary(tensor, name):
    if ':' in name:
        parts = name.split(':')
        assert len(parts) == 2 and parts[1] == '0'
        name = parts[0]
    with tf.name_scope(name):
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('has_nan', tf.cast(tf.reduce_any(tf.is_nan(tensor)), tf.int32))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))


class SummaryLogger:
    def __init__(self, log_dir=None, name=None, log_writer=None, graph=None):
        if log_writer:
            self.log_writer = log_writer
            return
        elif not log_dir:
            raise ValueError('log_dir or log_writer must be provided')

        if name:
            if not NAME_PATTERN.match(name):
                raise ValueError(
                    'Invalid name "%s"! (should be a valid directory)' % name)
            log_dir = os.path.join(log_dir, name)
        self.log_writer = tf.summary.FileWriter(log_dir, graph=graph)
        self.acc_fp = open(
            os.path.join(log_dir,
                         ('%s-' % name if name else '') + 'confusion.pkl'),
            'ab'
        )

    class StepLogger:
        def __init__(self, logger, step):
            self.logger = logger  # type: SummaryLogger
            self.step = step
            self.summary = None
            self.confusions = {}

        def __enter__(self):
            self.summary = tf.Summary()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.log_writer.add_summary(self.summary, self.step)
            self.logger.log_writer.flush()
            # pickle.dump((self.step, self.confusions), self.logger.acc_fp)
            self.summary = None

        def log_scalar(self, name, value):
            s_val = self.summary.value.add()
            s_val.tag = name
            s_val.simple_value = value

        def log_flat_histogram(self, name, bucket_values):
            s_val = self.summary.value.add()
            s_val.tag = name
            hist = s_val.histo

            bucket_values = list(bucket_values)
            hist.min = float(0)
            hist.max = float(len(bucket_values) - 1)
            # We treat each value in the input vector as a single value in the
            # relevant bucket.
            hist.num = int(np.nansum(bucket_values))
            # To sum this, we actually need to multiply the count in each bucket
            # by the bucket value
            hist.sum = float(sum(i * v for i, v in enumerate(bucket_values) if not np.isnan(v)))
            hist.sum_squares = float(
                sum(i * i * v for i, v in enumerate(bucket_values) if not np.isnan(v)))
            for i, v in enumerate([0] + bucket_values + [0]):
                hist.bucket_limit.append(i - 0.5)
                hist.bucket.append(v if not np.isnan(v) else 0)
            pass

        def log_image(self, name, img):
            # Use BytesIO and not StringIO for saving an image in-memory. See
            # http://matplotlib.1069221.n5.nabble.com/savefig-and-StringIO-error-on-Python3-td44241.html
            import matplotlib.pyplot as plt
            img_str = io.BytesIO()
            plt.imsave(img_str, img, format='png')

            s_val = self.summary.value.add()
            s_val.tag = name
            s_val.image.CopyFrom(tf.Summary.Image(
                encoded_image_string=img_str.getvalue(),
                height=img.shape[0],
                width=img.shape[1]
            ))

        def log_multi(self, log_func, name, values, limit, flatten=False):
            if flatten and len(values) == 1 and limit >= 1:
                log_func(name, values[0])
                return

            for i, val in max_enumerate(values, limit):
                log_func('%s/%d' % (name, i + 1), val)

        def log_confusion(self, name, confusion, n_guesses=None, flatten=True):
            if type(confusion) is np.ndarray:
                if len(confusion.shape) == 2:
                    size, depth = confusion.shape[-1], 1
                    mat = np.expand_dims(confusion, axis=0)
                elif len(confusion.shape) == 3:
                    size, depth = confusion.shape[-1], confusion.shape[0]
                    mat = confusion
                else:
                    raise AssertionError()
                confusion = ConfusionMatrix(size, depth)
                confusion.matrix = mat

            if confusion.n == 0:
                return

            n_guesses = n_guesses if n_guesses else confusion.depth

            self.log_scalar('%s/num-unpredicted' % name,
                            np.sum(confusion.prediction_histogram[0] == 0))

            for i, layer in max_enumerate(confusion.matrix, n_guesses):
                self.log_image('%s/confusion/%d' % (name, i + 1), layer)

            self.log_flat_histogram('%s/input-hist' % name,
                                    confusion.input_histogram / confusion.n)

            self.log_multi(self.log_flat_histogram, '%s/prediction-hist' % name,
                           confusion.prediction_histogram / confusion.n,
                           n_guesses, flatten)

            self.log_multi(self.log_scalar, '%s/acc' % name,
                           confusion.acc, n_guesses, flatten)

            if confusion.depth > 1:
                self.log_multi(self.log_scalar, '%s/cum-acc' % name,
                               confusion.cumulative_acc, n_guesses, flatten)

            self.log_multi(self.log_weights, '%s/class-acc' % name,
                           confusion.class_acc, n_guesses, flatten)

            if confusion.depth > 1:
                self.log_multi(self.log_weights, '%s/cum-class-acc' % name,
                               confusion.cumulative_class_acc, n_guesses, flatten)

            self.confusions[name] = confusion.matrix.tolist()

        def log_weights(self, name, weights):
            weights = np.asarray(weights)
            max_w = np.nanmax(weights)
            if max_w == 0:
                max_w = 1  # Avoid NaN's
            if weights.ndim == 1:
                self.log_flat_histogram('%s/hist' % name, weights / max_w)
            elif weights.ndim == 2:
                self.log_image('%s/hist' % name, weights / max_w)
            self.log_scalar('%s/min' % name, np.nanmin(weights))
            self.log_scalar('%s/max' % name, np.nanmax(weights))
            self.log_scalar('%s/std' % name, np.nanstd(weights))
            self.log_scalar('%s/mean' % name, np.nanmean(weights))

        def log_summary(self, summary):
            self.logger.log_writer.add_summary(summary, self.step)

    def log_step(self, step):
        return SummaryLogger.StepLogger(self, step)

    def log_event(self, message, step=None, level=tf.LogMessage.INFO):
        event = tf.Event()
        event.wall_time = time.time()
        if step is not None:
            event.step = event
        event.log_message.level = level
        event.log_message.message = str(message)
        self.log_writer.add_event(event)

    def __del__(self):
        self.log_writer.close()
        self.acc_fp.close()


class NullSummaryLogger:
    def log_step(self, step):
        return None

    def log_event(self, *args, **kwargs):
        pass
