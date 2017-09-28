import os
import numpy as np
import tensorflow as tf

import c3d.classification
import c3d.pipeline.rasterizer
import c3d.shape.input
import c3d.util.terminal
import c3d.util.tfsaver

from collections import namedtuple
from c3d.shape.loss import SmartLoss


TrainStep = namedtuple('TrainStep', 'class_subset n_epochs')


# n_epochs:cls1,cls2,cls3,...
def parse_train_step(str_val):
    str_val = str_val.strip()
    if not ':' in str_val:
        n_epochs, subset = str_val, ''
    else:
        n_epochs, subset = str_val.split(':')
    if subset:
        subset = set(s.strip() for s in subset.split(','))
    else:
        subset = set()
    return TrainStep(n_epochs=int(n_epochs), class_subset=subset)


class SherdInput(c3d.shape.input.SherdDataset):
    def __init__(self, class_subset=None, *args, **kwargs):
        super(SherdInput, self).__init__(*args, **kwargs)
        self.class_subset = class_subset
        self.fracture_rasterizer = c3d.pipeline.rasterizer.Rasterizer(
            c3d.pipeline.rasterizer.FractureHandler(), 256, 256)

    @property
    def has_class_subset(self):
        return self.class_subset is not None and len(self.class_subset) > 0

    def file_id_filter(self, file_id):
        return not self.has_class_subset or file_id.label in self.class_subset

    def prepare_file(self, file_id):
        fracture = super(SherdInput, self).prepare_file(file_id)
        img = self.fracture_rasterizer.rasterize(fracture)
        # Images in PIL are row-major - https://stackoverflow.com/a/19016499
        return np.asarray(img, dtype=np.float).T / 255


class SherdImageInput(c3d.classification.ImageDataset):
    def __init__(self, class_subset=None, *args, **kwargs):
        super(SherdImageInput, self).__init__(*args, **kwargs)
        self.class_subset = class_subset

    @property
    def has_class_subset(self):
        return self.class_subset is not None and len(self.class_subset) > 0

    def file_id_filter(self, file_id):
        return not self.has_class_subset or file_id.label in self.class_subset


class Classifier(c3d.classification.FeatureClassifier):
    def __init__(self, dataset, summary_dir, cache_dir, train_steps=None, **kwargs):
        super(Classifier, self).__init__(dataset, **kwargs)
        self.cache_dir = cache_dir
        self.summary_dir = summary_dir

        # List of variables in the part that should be reset for each training
        # on a set of new classes. Practically, this holds the variables of the
        # last FC layer, and we reset these when training on a set of new
        # classes.
        self.retune_vars = []

        self.batch_input = tf.placeholder(tf.float32, [None, 256, 256])
        self.features_out = None
        self.features_in = None
        self.y = None

        self.build_model()

        self.labels = tf.placeholder(tf.int64, [None])
        self.smart_loss = SmartLoss(self.n_classes, self.y, self.labels)
        self.predicted_labels = self.smart_loss.predicted_labels

        self.loss = self.smart_loss.make_loss()
        self.acc = self.smart_loss.make_acc()

        if train_steps:
            # Prefer the train steps that were specified
            self.train_steps = [
                step if isinstance(step, TrainStep) else parse_train_step(step)
                for step in train_steps
            ]
        else:
            # If unspecified, just do a vanilla train over 10 epochs
            self.train_steps = [
                TrainStep(None, 10)
            ]

        self.train_step_index = tf.Variable(0, trainable=False,
                                            name='train_step_index')
        self.train_step_index_inc = self.train_step_index.assign_add(1)
        self.batch_num = tf.Variable(0, trainable=False,
                                     name='batch_num')
        self.batch_num_inc = self.batch_num.assign_add(1)

        self.saver = tf.train.Saver()

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
        l1 = tf.reshape(self.batch_input, [-1, 256, 256, 1])

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
        self.features_out = tf.reshape(h5b, [-1, 4096 * 2], name='features')

        # To enable feeding features directly, we wrap these with a placeholder
        # taking it's default value from the features_out layer.
        self.features_in = tf.placeholder_with_default(
            self.features_out, self.features_out.shape, name='features_in')

        with tf.name_scope('FC'):
            W = self.weights('W', [4096 * 2, self.n_classes])
            b = self.weights('b', [self.n_classes])
            self.retune_vars.extend([W, b])

            p = tf.matmul(self.features_in, W)
            softmax_p = tf.nn.softmax(p + b)

        self.y = softmax_p

    def prepare_retune(self):
        with tf.name_scope('FC'):
            self.tf_session.run(tf.variables_initializer(self.retune_vars))

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
            train_step = tf.train.AdadeltaOptimizer(0.2).minimize(self.loss)

            # Initialize all the variables.
            tf.global_variables_initializer().run()

            # Recover any previous run that may have been interrupted.
            self.load_last_run()

            c3d.util.terminal.configure_numpy_print()

            summary = c3d.classification.SummaryLogger(self.summary_dir)
            confusion = c3d.classification.ConfusionMatrix(self.n_classes, self.n_classes)

            for subset, n_epochs in self.train_steps[self.train_step_index.eval():]:
                print('************************************************')
                print('**************** Subset %s *********************' % subset)
                print('************************************************')
                self.dataset.class_subset = subset
                confusion.reset()
                self.smart_loss.reset(self.tf_session)

                for batch_ids, batch_imgs in self.dataset.files_batch_train_iter(100, n_epochs):
                    x = np.array(batch_imgs)
                    labels = np.asarray([self.label_to_index[b.label] for b in batch_ids])
                    batch_num, _, probs, predictions, acc = self.tf_session.run(
                        [self.batch_num_inc, train_step, self.y,
                         self.predicted_labels, self.acc], {
                            self.batch_input: x,
                            self.labels: labels,
                        })
                    all_predicted_labels = np.argsort(-probs)
                    confusion.record(labels, all_predicted_labels)
                    self.smart_loss.record(predictions, labels)

                    if batch_num % 5 == 0:
                        with summary.log_step(batch_num) as log:
                            log.log_scalar('train-batch-acc', acc)
                            print('%05d: %f' % (batch_num, acc))

                    if batch_num % 20 == 0:
                        # Update the weights and record. Note that these will
                        # only affect on the next step.
                        with summary.log_step(batch_num + 1) as log:
                            self.smart_loss.update_weights(self.tf_session, log)

                    if batch_num % 50 == 0:
                        self.saver.save(self.tf_session, os.path.join(self.cache_dir, 'model'),
                                        global_step=batch_num)
                        with summary.log_step(batch_num) as log:
                            log.log_confusion('train', confusion, n_guesses=3)
                            print(confusion.matrix[0])
                            print(confusion.acc[:3])

                        if subset and confusion.acc[0] > 0.71:
                            print('Stopping fine tune on ACC %s!' % confusion.cumulative_acc[:3])
                            self.prepare_retune()
                            break
                        confusion.reset()

                self.tf_session.run(self.train_step_index_inc)

    def _compute_features(self, inputs):
        return self.tf_session.run(self.features_out, {
            self.batch_input: np.array(inputs),
        })

    def _classify_features_to_all(self, features):
        return self.tf_session.run(self.y, {
            self.features_in: np.array(features),
        })

    # Implement pickle support
    def __getstate__(self):
        state = super(Classifier, self).__getstate__()
        state['model'] = c3d.util.tfsaver.export_model(self.tf_session)
        state['cache_dir'] = self.cache_dir
        state['summary_dir'] = self.summary_dir
        state['train_steps'] = self.train_steps
        return state

    def __setstate__(self, state):
        self.__init__(state['dataset'], state['summary_dir'], state['cache_dir'],
                      label_to_index=state['label_to_index'])
        c3d.util.tfsaver.import_model(self.tf_session, state['model'])
        self.cache_dir = state['cache_dir']
        self.train_steps = state['train_steps']
