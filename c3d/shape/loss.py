import numpy as np
import tensorflow as tf

import c3d.classification


class SmartLoss:
    def __init__(self, n_classes, predicted_probabilities, labels,
                 alpha=0.8, beta=5, eps=0.01, name='smartloss'):
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.name = name

        with tf.name_scope(name):
            self.input_label_weight_t = tf.Variable(
                tf.ones((n_classes,), dtype=tf.float32),
                trainable=False,
                name='input_label_weights')
            self.output_label_weight_t = tf.Variable(
                tf.ones((n_classes,), dtype=tf.float32),
                trainable=False,
                name='output_label_weights')

        self.confusion = c3d.classification.ConfusionMatrix(self.n_classes)

        self.predicted_probabilities = predicted_probabilities
        self.predicted_labels = tf.arg_max(predicted_probabilities, 1)
        self.labels = labels

    def make_loss(self):
        input_weight = tf.gather(self.input_label_weight_t, self.labels)
        output_weight = tf.gather(self.output_label_weight_t, self.predicted_labels)

        hit = tf.cast(tf.equal(self.predicted_labels, self.labels), 'float')
        loss_weight = input_weight * ((1 - hit) * output_weight + 1)

        return tf.reduce_mean(
            loss_weight * tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.labels, self.n_classes),
                logits=self.predicted_probabilities)
        )

    def make_acc(self):
        correct_prediction = tf.equal(self.predicted_labels, self.labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def record(self, predictions, labels):
        self.confusion.record(labels, predictions)

    def _f(self, x):
        return tf.exp(self.eps + self.beta * x)

    def _interpolate(self, start, end):
        return self.alpha * start + (1 - self.alpha) * end

    def reset(self, session):
        session.run([
            self.input_label_weight_t.assign(
                tf.ones((self.n_classes,), dtype=tf.float32),
            ),
            self.output_label_weight_t.assign(
                tf.ones((self.n_classes,), dtype=tf.float32),
            ),
        ])
        self.confusion.reset()

    def update_weights(self, session, log_step=None):
        # TODO - consider multi-choice accuracies to improve also upon not just
        # the first guess (may require doing a loss not only on first)
        false_accuracy = self.confusion.prediction_histogram[0] - self.confusion.diagonal[0]
        false_accuracy += 1 * (false_accuracy == 0)
        false_accuracy /= np.sum(false_accuracy)

        session.run([
            self.input_label_weight_t.assign(self._interpolate(
                self.input_label_weight_t,
                tf.cast(1 / self._f(self.confusion.class_acc[0]), tf.float32)
            )),
            self.output_label_weight_t.assign(self._interpolate(
                self.output_label_weight_t,
                tf.cast(self._f(self.confusion.class_acc[0]), tf.float32)
            )),
        ])

        if log_step:
            log_step.log_weights('%s/input_weights' % self.name,
                                 session.run(self.input_label_weight_t))
            log_step.log_weights('%s/output_weights' % self.name,
                                 session.run(self.output_label_weight_t))

        self.confusion.reset()
