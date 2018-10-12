import tensorflow as tf

from c3d.classification import ConfusionMatrix, TFConfusion


class SmartLoss:
    def __init__(self, n_classes, logits, labels,
                 gamma=0.8, alpha_input=2, alpha_output=5, use_fp=False, fake=False, normalize=True,
                 beta=0, reducer=tf.reduce_mean, out_always=False, name='smartloss'):
        with tf.name_scope(name):
            self.n_classes = n_classes
            self.gamma = gamma if not fake else 1
            self.alpha_input = alpha_input
            self.alpha_output = alpha_output
            self.use_fp = use_fp
            self.reducer = reducer
            self.out_always = out_always
            self.name = name
            self.logits = logits
            self.predicted_labels = tf.arg_max(logits, 1)
            self.confusion = TFConfusion(self.n_classes, 'confusion')
            self.old_confusion = TFConfusion(self.n_classes, 'old_confusion')
            self.normalize = normalize
            self.beta = beta

            if len(labels.shape) == 1:
                self.labels = labels
                self.labels_one_hot = tf.one_hot(self.labels, self.n_classes)
            elif len(labels.shape) == 2 and labels.shape[-1] == self.n_classes:
                self.labels_one_hot = labels
                self.labels = tf.arg_max(labels, 1)
            else:
                raise ValueError('Bad shape for labels ' + labels.shape)

            # Avoid int32 vs. int64 issues
            if self.predicted_labels.dtype != self.labels.dtype:
                self.predicted_labels = tf.cast(self.predicted_labels, self.labels.dtype)

            self.input_label_weight_t = tf.Variable(
                tf.ones((n_classes,), dtype=tf.float32),
                trainable=False,
                name='input_label_weights')
            self.output_label_weight_t = tf.Variable(
                tf.ones((n_classes,), dtype=tf.float32),
                trainable=False,
                name='output_label_weights')

            self.standalone_update_op = self.make_update_op()

            self.confusion_placeholder = tf.placeholder(
                self.confusion.matrix.dtype, self.confusion.matrix.shape,
                'confusion_placeholder'
            )
            self.set_confusion_op = self.confusion.make_assign_op(
                self.confusion_placeholder
            )

    def make_update_op(self):
        update_input_weight_op = self.input_label_weight_t.assign(
            self._moving_average(
                self.input_label_weight_t,
                self.new_input_weight())
        )
        update_output_weight_op = self.output_label_weight_t.assign(
            self._moving_average(
                self.output_label_weight_t,
                self.new_output_weight())
        )
        save_old_conf_op = self.old_confusion.make_copy_op(self.confusion)

        # Reset the confusion only after updating the weights and saving a copy
        pre_ops = [update_input_weight_op, update_output_weight_op, save_old_conf_op]
        with tf.control_dependencies(pre_ops):
            reset_conf_op = self.confusion.make_reset_op()

        # And return a group making sure all of this ran
        return tf.group(reset_conf_op, *pre_ops)

    def make_loss(self):
        input_weight = tf.gather(self.input_label_weight_t, self.labels)
        output_weight = tf.gather(self.output_label_weight_t, self.predicted_labels)

        hit = tf.cast(tf.equal(self.predicted_labels, self.labels), 'float')
        if self.out_always:
            loss_weight = input_weight * output_weight
        else:
            loss_weight = input_weight * (1 + (1 - hit) * output_weight)

        return self.reducer(
            loss_weight * tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_one_hot,
                logits=self.logits)
        )

    def _moving_average(self, start, end):
        # Where the 'end' values has nan values, we don't want to update. To
        # avoid the update. select only non-nan values
        return tf.where(
            tf.is_nan(end),
            start,
            self.gamma * start + (1 - self.gamma) * end
        )

    def reset(self, session):
        session.run([
            self.input_label_weight_t.assign(
                tf.ones((self.n_classes,), dtype=tf.float32),
            ),
            self.output_label_weight_t.assign(
                tf.ones((self.n_classes,), dtype=tf.float32),
            ),
            self.confusion.make_reset_op()
        ])

    def new_input_weight(self):
        class_acc = self.confusion.class_acc
        return self._mean1(tf.cast((
            tf.exp(self.beta * tf.cast(self.confusion.input_histogram, tf.float32)) *
            tf.exp(- self.alpha_input * class_acc)
        ), tf.float32))

    def _mean1(self, value):
        """Scale the mean of a non-negative tensor to 1."""
        if self.normalize:
            non_nan = tf.boolean_mask(value, tf.logical_not(tf.is_nan(value)))
            # If the entire Tensor is made of zeros/nans, this will return a
            # nan Tensor. This is OK, as we use the in the moving average on
            # the weights, and the moving average function ignores nan values.
            # An entire weight vector of zeros indeed means we evaluated nothing
            # and it's OK to ignore. An entire weight vector of nan's should not
            # happen.
            return value / tf.reduce_mean(non_nan)
        else:
            return value

    def new_output_weight(self):
        def false_positives():
            falses = self.confusion.false_positive_histogram
            sum_falses = tf.reduce_sum(falses)
            return tf.cond(
                sum_falses > 0,
                lambda: tf.cast(falses / sum_falses, tf.float32),
                # If the precision is 1.0, gradually shift weights to uniform 1.
                lambda: tf.ones((self.n_classes,), dtype=tf.float32),
            )

        def class_acc():
            return self.confusion.class_acc

        fp_or_acc = tf.cond(tf.constant(self.use_fp), false_positives, class_acc)
        return self._mean1(tf.cast(tf.exp(+ self.alpha_output * fp_or_acc), tf.float32))

    def log_reports(self, session, log_step=None):
        if log_step:
            input_w, output_w, old_conf = session.run([
                self.input_label_weight_t,
                self.output_label_weight_t,
                self.old_confusion.matrix,
            ])
            log_step.log_weights('%s/input_weights' % self.name, input_w)
            log_step.log_weights('%s/output_weights' % self.name, output_w)
            log_step.log_confusion('%s/confusion' % self.name, old_conf)

    def make_confusion_record_op(self, *args, **kwargs):
        return self.confusion.make_update_op(
            self.predicted_labels, self.labels, *args, **kwargs
        )

    def update_weights(self, session, log_step=None):
        session.run(self.standalone_update_op)
        self.log_reports(session, log_step)

    def set_confusion(self, session, data):
        if type(data) is ConfusionMatrix:
            data = data.matrix[0]
        session.run(self.set_confusion_op, feed_dict={
            self.confusion_placeholder: data
        })
