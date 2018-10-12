import os
import numpy as np
import tensorflow as tf

import c3d.classification
import c3d.smartloss
import c3d.util.terminal
import c3d.util.tfsaver
from c3d.util import imgutils
from .input import make_train_dataset


class Classifier(c3d.classification.FeatureClassifier):
    def __init__(self, dataset, resnet_dir, summary_dir, cache_dir, tf_session=None, n_epochs=70, batch_size=128):
        super(Classifier, self).__init__(dataset, tf_session)
        if not os.path.isdir(resnet_dir):
            raise ValueError('Invalid resnet dir %s' % resnet_dir)
        self.resnet_dir = resnet_dir
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.summary_dir = summary_dir
        self.cache_path = os.path.join(self.cache_dir, 'decor_feature_cache.pickle')
        self.temp_cache_path = os.path.join(self.cache_dir, 'decor_feature_cache.temp.pickle')

        self.resnet_features_out = None
        self.resnet_images_in = None

        self.batch_num = None

        self.optimizer = None
        self.saver = None

        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def resnet_model_file_path(self, n_layers, suffix):
        n_layers = str(n_layers)  # In case this was passed as an integer
        assert n_layers.isdigit()  # Avoid silly mistakes and path issues
        name = 'ResNet-L{n_layers}{suffix}'.format(
                n_layers=n_layers, suffix=suffix
        )
        return os.path.join(self.resnet_dir, name)

    def _compute_features(self, images):
        return self.tf_session.run(self.resnet_features_out, {
            self.resnet_images_in: imgutils.resnet_preprocess(images)
        })

    def _load_resnet(self):
        if self.resnet_features_out is not None:
            return

        saver = tf.train.import_meta_graph(
                self.resnet_model_file_path(50, '.meta'),
                clear_devices=True
        )
        saver.restore(
                self.tf_session, self.resnet_model_file_path(50, '.ckpt'))

        def get_output(name):
            return self.tf_session.graph.get_operation_by_name(name).outputs[0]

        # Take the latest features from each block, and average them on the
        # width and height of the image, to get a single feature vector from
        # each block per image in the batch
        f2im = get_output('scale2/block3/Relu')
        self.f2 = tf.reduce_mean(f2im, reduction_indices=[1, 2], name='avg_pool2')

        f3im = get_output('scale3/block4/Relu')
        self.f3 = tf.reduce_mean(f3im, reduction_indices=[1, 2], name='avg_pool3')

        f4im = get_output('scale4/block6/Relu')
        self.f4 = tf.reduce_mean(f4im, reduction_indices=[1, 2], name='avg_pool4')

        f5im = get_output('scale5/block3/Relu')
        self.f5 = tf.reduce_mean(f5im, reduction_indices=[1, 2], name='avg_pool5')

        # Now concatenate all the features from all the different blocks, along
        # dimension 1 (where 0 is the index in the batch).
        self.resnet_features_out = tf.concat([self.f2, self.f3, self.f4, self.f5], 1, 'concat_feat')

        # The input placeholder
        self.resnet_images_in = get_output('images')

    def _build_model(self):
        if self.resnet_features_out is not None:
            return
        tmp = set(tf.global_variables())
        self._load_resnet()
        self.resnet_vars = set(tf.global_variables()) - tmp

        self.batch_num = tf.Variable(
            0, trainable=False, name='batch_num'
        )

        self.batch_in = tf.placeholder(
            dtype=tf.uint8, shape=self.resnet_images_in.shape
        )

        self.batch_labels = tf.placeholder(dtype=tf.int64, shape=(None,), name='labels')
        self.labels_one_hot = tf.one_hot(self.batch_labels, self.n_classes, name='labels_one_hot')

        self.is_training = tf.placeholder_with_default(False, shape=(), name='is_train')

        tmp = set(tf.trainable_variables())
        net = tf.stop_gradient(self.resnet_features_out, 'stop_grads')
        if True:
            net = tf.layers.dropout(net, 0.8, training=self.is_training)
            net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
            net = tf.layers.dropout(net, 0.8, training=self.is_training)

        self.logits = tf.layers.dense(net, self.n_classes, activation=None, name='logits')
        self.y = tf.nn.softmax(self.logits)

        self.fine_tune_vars = list(
            set(tf.trainable_variables()) - tmp
        )

        self.smartloss = c3d.smartloss.SmartLoss(
            n_classes=self.n_classes,
            logits=self.logits, labels=self.batch_labels,
        )
        self.smartloss_update = self.smartloss.make_update_op()
        self.loss = self.smartloss.make_loss()

        self.predicted_labels = tf.arg_max(self.logits, 1)
        # Avoid int32 vs. int64 issues
        if self.predicted_labels.dtype != self.batch_labels.dtype:
            self.predicted_labels = tf.cast(self.predicted_labels, self.batch_labels.dtype)

        self.acc = tf.reduce_mean(
            tf.to_float(tf.equal(self.predicted_labels, self.batch_labels))
        )

        self.optimizer = tf.train.AdamOptimizer()
        self.saver = tf.train.Saver()

    def _prepare(self):
        super(Classifier, self)._prepare()
        self._build_model()

    def _classify_features_to_all(self, features):
        return self.tf_session.run(self.y, {
            self.resnet_features_out: np.asanyarray(features),
        })

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
        with self.tf_session.as_default():
            # Create the optimizer prior to initializing the variables, to also
            # initialize it's internal variables.
            trainable_variables = self.fine_tune_vars
            grads = tf.gradients(self.loss, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)

            with tf.control_dependencies([self.smartloss.make_confusion_record_op()]):
                train_step = self.optimizer.apply_gradients(
                    zip(grads, trainable_variables), name='train_step',
                    global_step=self.batch_num
                )

            # Initialize all the variables.
            tf.initialize_variables(
                set(tf.global_variables()) - self.resnet_vars
            ).run()

            # Recover any previous run that may have been interrupted.
            self.load_last_run()
            print('Loaded last')

            c3d.util.terminal.configure_numpy_print()

            summary = c3d.classification.SummaryLogger(self.summary_dir, graph=self.tf_session.graph)
            confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)

            def debug_tensor(name, v):
                print('Name: mean: %f, std %d, min %f, max %f' % (
                    np.mean(v), np.std(v), np.min(v), np.max(v)
                ))

            print('Load')
            batch_images, batch_labels = make_train_dataset(self.dataset, self.label_to_index, self.batch_size)
            while True:
                print('Prep')
                x, labels = self.tf_session.run([batch_images, batch_labels])
                debug_tensor('images', x)
                x_augment = imgutils.resnet_preprocess(x)
                debug_tensor('resnet images', x_augment)
                print('Run')
                batch_num, _, probs, predictions, acc, loss_c, f2, f3, f4, f5 = self.tf_session.run(
                    [self.batch_num, train_step, self.y,
                     self.predicted_labels, self.acc,
                     self.loss, self.f2, self.f3, self.f4, self.f5
                     ], {
                        self.resnet_images_in: x_augment,
                        self.batch_labels: labels,
                    })
                debug_tensor('f2', f2)
                debug_tensor('f3', f3)
                debug_tensor('f4', f4)
                debug_tensor('f5', f5)

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
                        print(np.sum(confusion.prediction_histogram != 0, axis=1))

                    confusion.reset()

    def test(self):
        raise NotImplementedError()

    # Implement pickle support
    def __getstate__(self):
        state = super(Classifier, self).__getstate__()
        state['model'] = c3d.util.tfsaver.export_model(self.tf_session)
        state['cache_dir'] = self.cache_dir
        state['summary_dir'] = self.summary_dir
        state['resnet_dir'] = self.resnet_dir
        state['batch_size'] = self.batch_size
        return state

    def __setstate__(self, state):
        self.__init__(
            dataset=state['dataset'],
            cache_dir=state['cache_dir'],
            summary_dir=state['summary_dir'],
            resnet_dir=state['resnet_dir'],
            batch_size=state['batch_size'],
        )
        self.label_to_index = state['label_to_index']
        self._build_model()
        c3d.util.tfsaver.import_model(self.tf_session, state['model'])
