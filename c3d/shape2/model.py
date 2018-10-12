import os
import numpy as np
import tensorflow as tf

import c3d.classification
import c3d.util.terminal
import c3d.util.tfsaver

from pointnet.models import pointnet_cls
import pointnet.train

import c3d.smartloss
import c3d.shape2.gen_pcl_dataset


LR_INIT = 0.001 * 0.001 # Hack
LR_DECAY_STEP = 20000
LR_DECAY_RATE = 0.7

BN_INIT_DECAY = 0.5
BN_DECAY_RATE = 0.5
BN_DECAY_STEP = LR_DECAY_STEP
BN_DECAY_CLIP = 0.99


class Classifier(c3d.classification.FeatureClassifier):
    def __init__(self, dataset, K, summary_dir, cache_dir, num_points=None,
                 batch_size=100, eval_data=None, **kwargs):
        super(Classifier, self).__init__(dataset, **kwargs)
        self.cache_dir = cache_dir
        self.summary_dir = summary_dir

        self.num_points = num_points
        self.K = K
        self.n_epochs = 7000

        self.batch_size = batch_size
        self.batch_num = tf.Variable(
            0, trainable=False, name='batch_num')

        self.batch_labels = tf.placeholder(tf.int64, [None])
        self.batch_input = tf.placeholder(tf.float32, [None, self.num_points, K])

        self.features_out = None
        self.features_in = None
        self.logits = None  # Unscaled logits
        self.y = None  # Logits after softmax (i.e. probabilities)

        self.lr = pointnet.train.get_learning_rate(
            self.batch_num, base_rate=LR_INIT,
            batch_size=batch_size, decay_step=LR_DECAY_STEP,
            decay_rate=LR_DECAY_RATE
        )
        self.bn_decay = pointnet.train.get_bn_decay(
            self.batch_num, base_rate=BN_INIT_DECAY, batch_size=batch_size,
            decay_step=BN_DECAY_STEP, decay_rate=BN_DECAY_RATE
        )
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.is_training = None
        self.predicted_labels = None

        self.build_model()

        use_smartloss = True
        self.smartloss = c3d.smartloss.SmartLoss(n_classes=self.n_classes,
                                                 logits=self.logits, labels=self.batch_labels,
                                                 fake=not use_smartloss)
        self.smartloss_update = self.smartloss.make_update_op()
        self.classify_loss = self.smartloss.make_loss()
        self.loss = self.classify_loss

        self.acc = tf.reduce_mean(
            tf.to_float(tf.equal(self.predicted_labels, self.batch_labels))
        )

        self.eval_data = eval_data
        self.saver = tf.train.Saver()

    @property
    def profile_names(self):
        return self.label_to_index.keys()

    def build_model(self):
        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

        features = pointnet_cls.get_model_multi_features(
            self.batch_input, names=('points', 'angles'), Ks=(4, 4),
            is_training=self.is_training, bn_decay=self.bn_decay,
            input_transformer=False,  # Network seems to behave strangely with this
            feature_transformer=False
        )

        self.features_out = tf.identity(features, name='features')

        # To enable feeding features directly, we wrap these with a placeholder
        # taking it's default value from the features_out layer.
        self.features_in = tf.placeholder_with_default(
            self.features_out, self.features_out.shape, name='features_in')

        self.logits = pointnet_cls.get_model_scores(
            self.features_in, self.is_training, self.n_classes, self.bn_decay
        )
        #self.logits /= tf.tile(
        #    tf.reduce_max(self.logits, axis=1, keep_dims=True),
        #    [1, self.n_classes]
        #) * 10
        self.y = tf.nn.softmax(self.logits)

        self.predicted_labels = tf.arg_max(self.logits, 1)
        # Avoid int32 vs. int64 issues
        if self.predicted_labels.dtype != self.batch_labels.dtype:
            self.predicted_labels = tf.cast(self.predicted_labels, self.batch_labels.dtype)

    def load_last_run(self):
        if not os.path.isdir(self.cache_dir):
            raise ValueError('Invalid cache dir %s' % self.cache_dir)
        try:
            # Attempt restoring previously interrupted training sessions
            if os.path.exists(os.path.join(self.cache_dir, 'checkpoint')):
                self.saver.restore(
                    self.tf_session,
                    tf.train.latest_checkpoint(self.cache_dir))
                return True
        except:
            # If restoration failed, it may be because of changes in the
            # model or other unknown reasons. Log and continue as if there
            # was nothing to restore.
            import traceback
            traceback.print_exc()
        return False

    def _train(self):
        print('Will train')
        with self.tf_session.as_default():
            # Create the optimizer prior to initializing the variables, to also
            # initialize it's internal variables.
            trainable_variables = tf.trainable_variables()
            grads = tf.gradients(self.loss, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            with tf.control_dependencies([self.smartloss.make_confusion_record_op()]):
                train_step = self.optimizer.apply_gradients(
                    zip(grads, trainable_variables), name='train_step',
                    global_step=self.batch_num
                )

            # Initialize all the variables.
            tf.global_variables_initializer().run()

            # Recover any previous run that may have been interrupted.
            self.load_last_run()
            print('Loaded last')

            c3d.util.terminal.configure_numpy_print()

            summary = c3d.classification.SummaryLogger(self.summary_dir)
            confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)
            eval_confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)

            for batch_ids, batch_points in self.dataset.files_batch_train_iter(self.batch_size, self.n_epochs):
                x = np.array(batch_points)
                labels = np.asarray([self.label_to_index[b.label] for b in batch_ids])
                batch_num, _, probs, predictions, acc, loss_c = self.tf_session.run(
                    [self.batch_num, train_step, self.y,
                     self.predicted_labels, self.acc,
                     self.classify_loss
                     ], {
                        self.batch_input: x,
                        self.batch_labels: labels,
                    })
                if np.any(np.isnan(loss_c)):
                    print('Stopping on NaN classification loss!')
                    break
                all_predicted_labels = np.argsort(-probs)
                confusion.record(labels, all_predicted_labels)

                if batch_num % 5 == 0:
                    with summary.log_step(batch_num) as log:
                        log.log_scalar('train-batch-acc', acc)
                        print('%05d: %f (loss_c: %f)' % (batch_num, acc, loss_c))

                if batch_num % 50 == 0:
                    self.saver.save(self.tf_session, os.path.join(self.cache_dir, 'model'),
                                    global_step=batch_num)
                    with summary.log_step(batch_num) as log:
                        self.smartloss.update_weights(self.tf_session, log)
                        log.log_confusion('train', confusion, n_guesses=6)
                        print(confusion.matrix[0])
                        print(confusion.acc[:6])
                        print(np.nanmean(confusion.class_acc, axis=1)[:6])
                        print(np.sum(confusion.prediction_histogram != 0, axis=1))
                    confusion.reset()

                if batch_num % 200 == 0 and self.eval_data:
                    for e_batch_ids, e_batch_points in self.eval_data.files_batch_iter(self.batch_size, num_epochs=1):
                        labels = np.asarray([self.label_to_index[b.label] for b in e_batch_ids])
                        probs = self.tf_session.run(self.y, {
                            self.batch_input: np.array(e_batch_points),
                        })
                        all_predicted_labels = np.argsort(-probs)
                        eval_confusion.record(labels, all_predicted_labels)
                    with summary.log_step(batch_num) as log:
                        log.log_confusion('eval', eval_confusion, n_guesses=6)
                        print(eval_confusion.matrix[0])
                        print(eval_confusion.acc[:6])
                        print(np.nanmean(eval_confusion.class_acc, axis=1)[:6])
                        print(np.sum(eval_confusion.prediction_histogram != 0, axis=1))
                    eval_confusion.reset()

    def _compute_features(self, inputs):
        return self.tf_session.run(self.features_out, {
            self.batch_input: np.asanyarray(inputs),
        })

    def _classify_features_to_all(self, features):
        return self.tf_session.run(self.y, {
            self.features_in: np.asanyarray(features),
        })

    # Implement pickle support
    def __getstate__(self):
        state = super(Classifier, self).__getstate__()
        state['model'] = c3d.util.tfsaver.export_model(self.tf_session)
        state['K'] = self.K
        state['cache_dir'] = self.cache_dir
        state['summary_dir'] = self.summary_dir
        return state

    def __setstate__(self, state):
        self.__init__(
            dataset=state['dataset'], K=state.get('K', c3d.shape2.gen_pcl_dataset.NUM_POINT_DIMENSIONS),
            summary_dir=state['summary_dir'], cache_dir=state['cache_dir'],
            label_to_index=state['label_to_index']
        )
        c3d.util.tfsaver.import_model(self.tf_session, state['model'])
        self.cache_dir = state['cache_dir']
