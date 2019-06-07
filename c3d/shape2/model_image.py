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


class ImageClassifier(c3d.classification.FeatureClassifier):
    def __init__(self, dataset, config: outlinenet.OutlineNetConfig,
                 summary_dir, cache_dir, eval_data=None, **kwargs):
        super(ImageClassifier, self).__init__(dataset, **kwargs)
        self.cache_dir = cache_dir
        self.summary_dir = summary_dir

        self.config = config
        self.config.set_n_classes(self.n_classes)
        self.outlinenet = c3d.shape2.outlinenet.OutlineNet(self.config)
        self.n_epochs = 7000
        self.log_guesses = 10

        self.batch_num = tf.Variable(0, trainable=False, name='batch_num')

        self.batch_labels = None
        self.batch_images = None

        self.features = None
        self.y = None  # Logits after softmax (i.e. probabilities)

        self.lr = self.outlinenet.make_lr(self.batch_num)
        self.optimizer = self.outlinenet.get_optimizer(self.lr)

        self.is_training = None
        self.predicted_labels = None
        self.debug = {}
        self.logits = None

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

    def weights(self, name, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

    def conv2d(self, name, input, filter_shape, strides=[1, 1, 1, 1], non_linearity=None):
        with tf.name_scope(name):
            w = self.weights('W', filter_shape)
            result = tf.nn.conv2d(input, w, strides=strides, padding='SAME')
            if non_linearity:
                b = self.weights('b', [filter_shape[-1]])
                result = non_linearity(result + b)
        return result

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def build_model(self):
        self.batch_labels = tf.placeholder(tf.int64, [None], name='batch_labels')
        self.batch_images = tf.placeholder(tf.float32, [None, 256, 256], name='batch_images')

        l1 = tf.reshape(self.batch_images, [-1, 256, 256, 1])

        h1 = self.conv2d('h1', l1, [5, 5, 1, 16], non_linearity=tf.nn.relu)
        h1b = self.max_pool_2x2(h1)

        h2 = self.conv2d('h2', h1b, [3, 3, 16, 32], non_linearity=tf.nn.relu)
        h2b = self.max_pool_2x2(h2)

        h3 = self.conv2d('h3', h2b, [3, 3, 32, 32], non_linearity=tf.nn.relu)
        h3b = self.max_pool_2x2(h3)

        h4 = self.conv2d('h4', h3b, [3, 3, 32, 64], non_linearity=tf.nn.relu)
        h4b = self.max_pool_2x2(h4)

        h5 = self.conv2d('h5', h4b, [3, 3, 64, 128], non_linearity=tf.nn.relu)
        h5b = self.max_pool_2x2(h5)

        # Flattening all the data after the convolutions gives us the features.
        self.features = tf.reshape(h5b, [-1, 4096 * 2], name='features')

        with tf.name_scope('FC'):
            W = self.weights('W', [4096 * 2, self.n_classes])
            b = self.weights('b', [self.n_classes])

            p = tf.matmul(self.features, W)
            self.logits = p + b

        self.y = tf.nn.softmax(self.logits)

        self.is_training = tf.placeholder_with_default(
            False, shape=(), name='is_training'
        )

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

    def _eval_on(self, name, batch_iter, confusion,
                 log_step: c3d.classification.summary.SummaryLogger.StepLogger,
                 n_batches=None):
        for i, (e_batch_ids, e_batch_images) in enumerate(batch_iter):
            if n_batches is not None and i > n_batches:
                break
            labels = np.asarray([self.label_to_index[b.label] for b in e_batch_ids])
            probs = self.tf_session.run(self.y, N({
                self.batch_images: e_batch_images,
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

            for batch_ids, batch_images in self.dataset.files_batch_train_iter(self.batch_size, self.n_epochs):
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
                        self.batch_images: batch_images,
                        self.batch_labels: labels,
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
        return self.tf_session.run(self.features, N({
            self.batch_images: inputs,
        }))

    def _classify_features_to_all(self, features):
        return self.tf_session.run(self.y, {
            self.features: np.asanyarray(features),
        })

    # Implement pickle support
    def __getstate__(self):
        state = super(ImageClassifier, self).__getstate__()
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
            summary_dir=state['summary_dir'],
            cache_dir=state['cache_dir'],
            label_to_index=state['label_to_index']
        )
        c3d.util.tfsaver.import_model(self.tf_session, state['model'])
