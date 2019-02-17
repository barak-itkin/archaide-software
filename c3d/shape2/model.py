import os
import numpy as np
import pickle
import tensorflow as tf

import c3d.classification
import c3d.util.terminal
import c3d.util.tfsaver
from c3d.classification.summary import tensor_summary
from c3d.shape2 import outlinenet

import c3d.smartloss
import c3d.shape2.gen_pcl_dataset
import c3d.shape2.tf_simple_conv
import c3d.shape2.outlinenet


def N(feed_dict):
    return {
        k: v for k, v in feed_dict.items()
        if k is not None and v is not None
    }


class Classifier(c3d.classification.FeatureClassifier):
    def __init__(self, dataset, config: outlinenet.OutlineNetConfig,
                 summary_dir, cache_dir, eval_data=None, **kwargs):
        super(Classifier, self).__init__(dataset, **kwargs)
        self.cache_dir = cache_dir
        self.summary_dir = summary_dir

        self.config = config
        self.config.set_n_classes(self.n_classes)
        self.outlinenet = c3d.shape2.outlinenet.OutlineNet(self.config)
        self.n_epochs = 7000
        self.log_guesses = 10

        self.batch_num = tf.Variable(0, trainable=False, name='batch_num')

        self.batch_labels = None
        self.batch_points = None
        self.batch_groups = None
        self.batch_counts = None

        self.features = None
        self.y = None  # Logits after softmax (i.e. probabilities)

        self.lr = self.outlinenet.make_lr(self.batch_num)
        self.optimizer = self.outlinenet.get_optimizer(self.lr)

        self.is_training = None
        self.predicted_labels = None
        self.debug = {}
        self.logits = None
        self.flat_logits = None

        self.build_model()

        self.smartloss, self.classify_loss = self.outlinenet.make_losses(
            batch_labels=self.batch_labels, logits=self.logits
        )
        self.reg_loss = self.outlinenet.get_reg_loss()
        self.loss = self.classify_loss + self.reg_loss

        self.acc = tf.reduce_mean(
            tf.to_float(tf.equal(self.predicted_labels, self.batch_labels))
        )

        self.eval_data = eval_data
        self.saver = tf.train.Saver()

    @property
    def batch_size(self):
        return self.config.learning_spec.batch_size

    @property
    def profile_names(self):
        return self.label_to_index.keys()

    def build_model(self):
        self.batch_labels = tf.placeholder(tf.int64, [None], name='batch_labels')
        self.batch_groups = tf.placeholder(tf.float32, [None, None], name='batch_groups')
        self.batch_points = tf.placeholder(tf.float32, [None, None, 2], name='batch_points')
        self.batch_counts = tf.placeholder(tf.int32, [None], name='batch_counts')

        self.is_training = tf.placeholder_with_default(
            False, shape=(), name='is_training'
        )

        self.features, self.logits, self.flat_logits = self.outlinenet.build(
            batch_points=self.outlinenet.augment_train_points(
                batch_points=self.batch_points, is_training=self.is_training
            ),
            batch_counts=self.batch_counts,
            batch_groups=self.batch_groups,
            is_training=self.is_training,
            debug=self.debug
        )

        self.y = tf.nn.softmax(self.flat_logits)

        self.predicted_labels = tf.arg_max(self.flat_logits, 1)
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

    def _get_pts_and_groups(self, batch_points):
        batch_points, batch_groups, batch_counts = (
            [p[0] for p in batch_points],
            [p[1] for p in batch_points],
            [p[2] for p in batch_points],
        )
        batch_points = np.array(batch_points)
        return batch_points, batch_groups, batch_counts

    def _eval_on(self, name, batch_iter, confusion,
                 log_step: c3d.classification.summary.SummaryLogger.StepLogger,
                 n_batches=None):
        for i, (e_batch_ids, e_batch_points) in enumerate(batch_iter):
            if n_batches is not None and i > n_batches:
                break
            e_batch_points, e_batch_groups, e_batch_counts = self._get_pts_and_groups(e_batch_points)
            labels = np.asarray([self.label_to_index[b.label] for b in e_batch_ids])
            probs = self.tf_session.run(self.y, N({
                self.batch_points: e_batch_points,
                self.batch_groups: e_batch_groups,
                self.batch_counts: e_batch_counts,
            }))
            all_predicted_labels = np.argsort(-probs)
            confusion.record(labels, all_predicted_labels)

        log_step.log_confusion(name, confusion, n_guesses=self.log_guesses)
        print('*********** %s **************' % name)
        print(confusion.matrix[0])
        print(confusion.acc[:self.log_guesses])
        print(np.nanmean(confusion.class_acc, axis=1)[:self.log_guesses])
        print(np.sum(confusion.prediction_histogram != 0, axis=1))
        confusion.reset()

    def _train(self):
        print('Will train')
        with self.tf_session.as_default():
            # Create the optimizer prior to initializing the variables, to also
            # initialize it's internal variables.
            trainable_variables = tf.trainable_variables()
            grads = tf.gradients(
                self.loss,
                trainable_variables
            )
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            deps = (
                [self.smartloss.make_confusion_record_op()]
                if self.smartloss is not None
                else []
            )
            with tf.control_dependencies(deps):
                train_step = self.optimizer.apply_gradients(
                    zip(grads, trainable_variables), name='train_step',
                    global_step=self.batch_num
                )

            with tf.name_scope('var-summaries'):
                for var in trainable_variables:
                    tensor_summary(var, var.name)

            with tf.name_scope('grad-summaries'):
                for g, var in zip(grads, trainable_variables):
                    tensor_summary(g, var.name)

            with tf.name_scope('layer-summaries'):
                for name, op in self.debug.items():
                    if isinstance(op, c3d.shape2.tf_simple_conv.Outlines):
                        op = op.points
                    tensor_summary(op, name)

            summary_period = 10
            all_summaries = tf.summary.merge_all()
            no_summary = tf.constant(False)

            def optional_summary():
                if (self.tf_session.run(self.batch_num) + 1) % summary_period == 0:
                    return all_summaries
                else:
                    return no_summary

            # Initialize all the variables.
            tf.global_variables_initializer().run()

            # Recover any previous run that may have been interrupted.
            self.load_last_run()
            print('Loaded last')

            c3d.util.terminal.configure_numpy_print()

            summary = c3d.classification.SummaryLogger(self.summary_dir, graph=self.tf_session.graph)
            confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)
            eval_confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)
            test_confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)

            test_set = pickle.loads(pickle.dumps(self.dataset))
            test_set.eval_mode = True
            test_set.do_caching = False
            test_iter = test_set.files_batch_test_iter(self.batch_size)

            stop_on_first = len(os.environ.get('C3D_INSPECT', '')) > 0

            for batch_ids, batch_points in self.dataset.files_batch_train_iter(self.batch_size, self.n_epochs):
                batch_points, batch_groups, batch_counts = self._get_pts_and_groups(batch_points)
                labels = np.asarray([self.label_to_index[b.label] for b in batch_ids])
                if stop_on_first:
                    import ipdb
                    ipdb.set_trace()
                    exit()
                batch_num, _, probs, predictions, acc, loss_c, maybe_summary = self.tf_session.run(
                    [self.batch_num, train_step, self.y,
                     self.predicted_labels, self.acc,
                     self.classify_loss, optional_summary()
                     ], N({
                        self.batch_points: batch_points,
                        self.batch_labels: labels,
                        self.batch_groups: batch_groups,
                        self.batch_counts: batch_counts,
                        self.is_training: True
                    })
                )

                if np.any(np.isnan(loss_c)):
                    print('Stopping on NaN classification loss!')
                    break
                all_predicted_labels = np.argsort(-probs)
                confusion.record(labels, all_predicted_labels)

                if batch_num % 5 == 0:
                    with summary.log_step(batch_num) as log:
                        log.log_scalar('train-batch-acc', acc)
                        print('%05d: %f (loss_c: %f)' % (batch_num, acc, loss_c))

                if maybe_summary:
                    with summary.log_step(batch_num) as log:
                        log.log_summary(maybe_summary)

                if batch_num % 50 == 0:
                    self.saver.save(self.tf_session, os.path.join(self.cache_dir, 'model'),
                                    global_step=batch_num)

                if batch_num % self.config.learning_spec.smartloss_accum_batches == 0:
                    with summary.log_step(batch_num) as log:
                        if self.smartloss is not None:
                            self.smartloss.update_weights(self.tf_session, log)
                        log.log_confusion('train', confusion, n_guesses=self.log_guesses)
                        print(confusion.matrix[0])
                        print(confusion.acc[:self.log_guesses])
                        print(np.nanmean(confusion.class_acc, axis=1)[:self.log_guesses])
                        print(np.sum(confusion.prediction_histogram != 0, axis=1))
                    confusion.reset()

                if batch_num % 200 == 0 and self.eval_data:
                    with summary.log_step(batch_num) as log:
                        self._eval_on(
                            name='eval',
                            batch_iter=self.eval_data.files_batch_iter(self.batch_size, num_epochs=1),
                            confusion=eval_confusion,
                            log_step=log
                        )

                if batch_num % 400 == 0:
                    with summary.log_step(batch_num) as log:
                        self._eval_on(
                            name='test', batch_iter=test_iter,
                            confusion=test_confusion, log_step=log,
                            n_batches=5
                        )

    def _compute_features(self, inputs):
        batch_points, batch_groups, batch_counts = self._get_pts_and_groups(inputs)
        return self.tf_session.run(self.features, N({
            self.batch_points: np.asanyarray(batch_points),
            self.batch_groups: batch_groups,
            self.batch_counts: batch_counts
        }))

    def _classify_features_to_all(self, features):
        return self.tf_session.run(self.y, {
            self.features: np.asanyarray(features),
        })

    # Implement pickle support
    def __getstate__(self):
        state = super(Classifier, self).__getstate__()
        state['config'] = self.config
        state['model'] = c3d.util.tfsaver.export_model(self.tf_session)
        state['cache_dir'] = self.cache_dir
        state['summary_dir'] = self.summary_dir
        return state

    def __setstate__(self, state):
        config = outlinenet.OutlineNetConfig()
        config.from_json(state['config'].to_json())
        self.__init__(
            dataset=state['dataset'], config=config,
            summary_dir=state['summary_dir'], cache_dir=state['cache_dir'],
            label_to_index=state['label_to_index']
        )
        c3d.util.tfsaver.import_model(self.tf_session, state['model'])
