# Partially based on https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import tensorflow as tf
import time

NAME_PATTERN = re.compile(r'^[\w\-_\.]+$')


def max_enumerate(values, limit):
    return zip(range(limit), values)


class SummaryLogger:
    def __init__(self, log_dir, name=None):
        if name:
            if not NAME_PATTERN.match(name):
                raise ValueError(
                    'Invalid name "%s"! (should be a valid directory)' % name)
            log_dir = os.path.join(log_dir, name)
        self.log_writer = tf.summary.FileWriter(log_dir)

    class StepLogger:
        def __init__(self, logger, step):
            self.logger = logger
            self.step = step
            self.summary = None

        def __enter__(self):
            self.summary = tf.Summary()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.log_writer.add_summary(self.summary, self.step)
            self.logger.log_writer.flush()
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
            hist.num = int(np.sum(bucket_values))
            # To sum this, we actually need to multiply the count in each bucket
            # by the bucket value
            hist.sum = float(sum(i * v for i, v in enumerate(bucket_values)))
            hist.sum_squares = float(
                sum(i * i * v for i, v in enumerate(bucket_values)))
            for i, v in enumerate([0] + bucket_values + [0]):
                hist.bucket_limit.append(i - 0.5)
                hist.bucket.append(v)
            pass

        def log_image(self, name, img):
            # Use BytesIO and not StringIO for saving an image in-memory. See
            # http://matplotlib.1069221.n5.nabble.com/savefig-and-StringIO-error-on-Python3-td44241.html
            img_str = io.BytesIO()
            plt.imsave(img_str, img, format='png')

            s_val = self.summary.value.add()
            s_val.tag = name
            s_val.image.CopyFrom(tf.Summary.Image(
                encoded_image_string=img_str.getvalue(),
                height=img.shape[0],
                width=img.shape[1]
            ))

        def log_multi(self, log_func, name, values, limit):
            for i, val in max_enumerate(values, limit):
                log_func('%s/%d' % (name, i + 1), val)

        def log_confusion(self, name, confusion, n_guesses=None):
            n_guesses = n_guesses if n_guesses else confusion.depth

            for i, layer in max_enumerate(confusion.matrix, n_guesses):
                self.log_image('%s/confusion/%d' % (name, i + 1), layer)

            self.log_flat_histogram('%s/input-hist' % name,
                                    confusion.input_histogram / confusion.n)

            self.log_multi(self.log_flat_histogram, '%s/prediction-hist' % name,
                           confusion.prediction_histogram / confusion.n,
                           n_guesses)

            self.log_multi(self.log_scalar, '%s/acc' % name,
                           confusion.acc, n_guesses)

            self.log_multi(self.log_scalar, '%s/cum-acc' % name,
                           confusion.cumulative_acc, n_guesses)

            self.log_multi(self.log_flat_histogram, '%s/class-acc' % name,
                           confusion.class_acc, n_guesses)

            self.log_multi(self.log_flat_histogram, '%s/cum-class-acc' % name,
                           confusion.cumulative_class_acc, n_guesses)

        def log_weights(self, name, weights):
            weights = np.asarray(weights)
            if weights.ndim == 1:
                self.log_flat_histogram('%s/hist' % name, weights / np.max(weights))
            elif weights.ndim == 2:
                self.log_image('%s/hist' % name, weights / np.max(weights))
            self.log_scalar('%s/min' % name, np.min(weights))
            self.log_scalar('%s/max' % name, np.max(weights))
            self.log_scalar('%s/std' % name, np.std(weights))
            self.log_scalar('%s/mean' % name, np.mean(weights))

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
