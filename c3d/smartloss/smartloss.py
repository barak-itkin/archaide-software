import tensorflow as tf

from c3d.classification import TFConfusion


class SmartLoss:
    def __init__(self, n_classes, logits, labels,
                 gamma=0.8, alpha_input=2, alpha_output=5, use_fp=False, fake=False, normalize=True,
                 reducer=tf.reduce_mean, out_always=False, name='smartloss', hit_k=1, input_k=1, output_k=1,
                 use_in=True, use_out=True, out_shift=1, focal_loss=False, focal_gamma=2):
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
            self.labels = labels
            self.hit_k = hit_k
            self.input_k = input_k
            self.output_k = output_k
            self.use_in = use_in if not fake else False
            self.use_out = use_out if not fake else False
            self.out_shift = out_shift
            self.focal_loss = focal_loss
            self.focal_gamma = focal_gamma

            _, self.top_predicted_labels = tf.nn.top_k(logits, sorted=True, k=self.n_classes)
            self.normalize = normalize

            assert len(self.labels.shape) + 1 == len(self.logits.shape)

            # Avoid int32 vs. int64 issues
            if self.top_predicted_labels.dtype != self.labels.dtype:
                self.top_predicted_labels = tf.cast(self.top_predicted_labels, self.labels.dtype)
            self.predicted_labels = self.top_predicted_labels[..., 0]
            self.confusion = TFConfusion(self.n_classes, 'confusion', depth=self.n_classes,
                                         dtype=self.labels.dtype)

            self.input_label_weight_t = tf.Variable(
                tf.ones((n_classes,), dtype=tf.float32),
                trainable=False,
                name='input_label_weights')
            self.output_label_weight_t = tf.Variable(
                tf.ones((n_classes,), dtype=tf.float32),
                trainable=False,
                name='output_label_weights')

            self.standalone_update_op = self.make_update_op()

    def make_update_op(self):
        update_ops = []
        if self.use_in:
            update_input_weight_op = self.input_label_weight_t.assign(
                self._moving_average(
                    self.input_label_weight_t,
                    self.new_input_weight())
            )
            update_ops.append(update_input_weight_op)

        if self.use_out:
            update_output_weight_op = self.output_label_weight_t.assign(
                self._moving_average(
                    self.output_label_weight_t,
                    self.new_output_weight())
            )
            update_ops.append(update_output_weight_op)

        # Reset the confusion only after updating the weights
        with tf.control_dependencies(update_ops):
            reset_conf_op = self.confusion.make_reset_op()

        # And return a group making sure all of this ran
        return tf.group(reset_conf_op, *update_ops)

    def make_loss(self):
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(self.labels), tf.shape(self.predicted_labels))
        ]):
            if self.use_in:
                input_weight = tf.gather(self.input_label_weight_t,
                                         indices=self.labels)
            else:
                input_weight = tf.ones_like(
                    self.labels, dtype=self.input_label_weight_t.dtype
                )

            if self.use_out:
                output_weight = tf.gather(self.output_label_weight_t,
                                          indices=self.predicted_labels)
            else:
                output_weight = tf.ones_like(
                    self.labels, dtype=self.output_label_weight_t.dtype
                )

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(self.labels),
                            tf.shape(self.top_predicted_labels)[:-1])
        ]):
            hit = tf.cast(
                tf.reduce_any(
                    tf.equal(self.top_predicted_labels[..., :self.hit_k],
                             tf.expand_dims(self.labels, axis=-1)),
                    axis=-1
                ),
                'float'
            )

        if self.use_out and not self.out_always:
            output_weight = (self.out_shift + (1 - hit) * output_weight)
            output_weight = self._mean1(output_weight)

        loss_weight = input_weight * output_weight

        if self.focal_loss:
            if len(self.labels.shape) != 1:
                raise NotImplementedError('Focal loss not implemented with multi-dimensional labels')

            with tf.name_scope('focal_loss'):
                probs = tf.nn.softmax(self.logits, axis=-1)
                loss_weight *= tf.pow(
                    1 - tf.gather(probs, indices=self.labels, axis=-1),
                    self.focal_gamma
                )

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(self.labels), tf.shape(loss_weight))
        ]):
            return self.reducer(
                loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels,
                    logits=self.logits
                )
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
        class_acc = self.confusion.cum_class_acc[self.input_k]
        return self._mean1(tf.cast((
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
            falses = self.confusion.false_positive_histogram[0]
            sum_falses = tf.reduce_sum(falses)
            return tf.cond(
                sum_falses > 0,
                lambda: tf.cast(falses / sum_falses, tf.float32),
                # If the precision is 1.0, gradually shift weights to uniform 1.
                lambda: tf.ones((self.n_classes,), dtype=tf.float32),
            )

        def class_acc():
            return self.confusion.cum_class_acc[self.output_k]

        fp_or_acc = tf.cond(tf.constant(self.use_fp), false_positives, class_acc)
        return self._mean1(tf.cast(tf.exp(+ self.alpha_output * fp_or_acc), tf.float32))

    def log_reports(self, session, log_step=None):
        if log_step:
            input_w, output_w = session.run([
                self.input_label_weight_t,
                self.output_label_weight_t,
            ])
            log_step.log_weights('%s/input_weights' % self.name, input_w)
            log_step.log_weights('%s/output_weights' % self.name, output_w)

    def make_confusion_record_op(self):
        return self.confusion.make_update_op(self.top_predicted_labels, self.labels)

    def update_weights(self, session, log_step=None):
        session.run(self.standalone_update_op)
        self.log_reports(session, log_step)
