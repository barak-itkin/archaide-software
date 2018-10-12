import argparse
import json
import logging
import numpy as np
import os
import pickle
import sys

from IPython import embed
from c3d.classification import ConfusionMatrix
from .model import Classifier
from .gen_pcl_dataset import SherdSVGDataset, SherdDataset, NUM_POINT_DIMENSIONS

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(stream=sys.stdout, format=FORMAT, level=logging.DEBUG)

TRAIN = 'train'
TEST = 'test'
CLASSIFY = 'classify'
EVAL = 'eval'
ACTIONS = [TRAIN, TEST, CLASSIFY, EVAL]


def make_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Fracture shape based classifier')

    parser.add_argument('model_path', type=str, metavar='model-path',
                        help='Path for saving/loading the model file')
    parser.add_argument('data_dir', type=str, metavar='data-dir',
                        help='Path to folder containing the data')

    parser.add_argument('--svg-inputs', default=False, action='store_true',
                        help='Specify the data dir contains fracture SVG files '
                        '(and not fracture JSON files)')
    parser.add_argument('--disable-tensorflow-logs', default=False,
                        action='store_true', help='Hide logging by TensorFlow')
    parser.add_argument('--k', type=int, default=1, required=False,
                        help='Number of best classes to predict')
    parser.add_argument('--train-to-test-ratio', type=int, default=6,
                        required=False,
                        help='The ratio (num of train samples) / (num of '
                        'test samples)')
    parser.add_argument('--label-mapping-file', default=None, type=str,
                        help='Path to a JSON mapping file, mapping type names to '
                        'new types')
    parser.add_argument('--loose-mapping', default=False, action='store_true',
                        help='When type mapping is specified, allow profiles not present'
                        'in the mapping. By default, profiles not present in the mapping will'
                        'be ignored')
    parser.add_argument('--regular_y', action='store_true', default=False,
                        help='Is the top of the vessel at higher Y? (not the default)')

    actions = parser.add_subparsers(dest='action')

    train_parser = actions.add_parser(TRAIN, help='Train the classifier')
    train_parser.add_argument('--cache_dir', type=str,
                              default='cache', required=False,
                              help="Directory for caching models during the "
                              "train (in case it's interrupted)")
    train_parser.add_argument('--summary_dir', type=str,
                              default='summary', required=False,
                              help='Directory for TensorBoard summary during '
                              'the train')
    train_parser.add_argument('--no_smartloss', default=False, action='store_true',
                              help='Do not use smartloss for the training')
    train_parser.add_argument('--eval_set', type=str, default=None,
                              help='If specified, use the data in this directory '
                                   'for evaluation during the training (test set)')

    test_parser = actions.add_parser(TEST,
                                     help='Evaluate the training process. Use '
                                     'ONLY with the same data folder as for '
                                     'the training (and tehe same '
                                     '--train-to-test-ratio)')

    classify_parser = actions.add_parser(CLASSIFY,
                                         help='Classify given inputs')

    eval_parser = actions.add_parser(EVAL,
                                     help='Load model and start IPython shell')

    return parser


args = make_parser().parse_args()
train_test_ratio = args.train_to_test_ratio if args.action != CLASSIFY else None

if args.disable_tensorflow_logs:
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    # And also silence the native code
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


elif args.svg_inputs:
    data = SherdSVGDataset(data_root=args.data_dir, regular_y=args.regular_y, train_to_test_ratio=train_test_ratio)
else:
    data = SherdDataset(data_root=args.data_dir, train_to_test_ratio=train_test_ratio)

if args.label_mapping_file:
    with open(args.label_mapping_file, 'r') as f:
        data.label_map = json.load(f)
    data.loose_label_map = args.loose_mapping

data.enable_cache()

if args.action == TRAIN:
    data.do_noise = True
    data.balance = True

    if args.eval_set:
        eval_data = SherdSVGDataset(data_root=args.eval_set, regular_y=args.regular_y)
        eval_data.do_noise = False
        eval_data.balance = False
        eval_data.label_map = data.label_map
        eval_data.loose_label_map = data.loose_label_map
    else:
        eval_data = None

    c = Classifier(
        data, K=NUM_POINT_DIMENSIONS,
        summary_dir=args.summary_dir, cache_dir=args.cache_dir,
        eval_data=eval_data
    )
    try:
        c.train()
    except (KeyboardInterrupt, InterruptedError) as e:
        logging.warning('Training interrupted, saving our current model')
    with open(args.model_path, 'wb') as f:
        pickle.dump(c, f)

else:
    data.do_noise = False
    data.balance = False
    with open(args.model_path, 'rb') as f:
        c = pickle.load(f)
    c.dataset = data

    def classify(files_batch_iter, keep_indices=False):
        for batch_ids, batchs_imgs in files_batch_iter:
            batch_prediction_indices = c.predict_top_k(batchs_imgs, args.k)
            if keep_indices:
                yield batch_ids, batch_prediction_indices
            else:
                yield batch_ids, [[c.index_to_label[i] for i in prediction_indices]
                                  for prediction_indices in batch_prediction_indices]

    if args.action == CLASSIFY:
        confusion = ConfusionMatrix(c.n_classes, args.k)
        for batch_ids, batch_predictions in classify(
                data.files_batch_iter(batch_size=100, num_epochs=1), keep_indices=True):
            batch_labels = [c.label_to_index[f.label] for f in batch_ids]
            confusion.record(batch_labels, batch_predictions)
            for file_id, predictions in zip(batch_ids, batch_predictions):
                # Intentionally print and don't log, so that the format is easy to
                # expect and parse.
                print(data.file_path(file_id))
                for i, p in enumerate(predictions):
                    print('%3d: %s' % (i + 1, c.index_to_label[p]))
        logging.info('Total:\n%s', confusion.n)
        logging.info('Accuracy:\n%s', confusion.acc)
        logging.info('Cumulative accuracy:\n%s', confusion.cumulative_acc)
        logging.info('Class Accuracy:\n%s', np.nanmean(confusion.class_acc, axis=1))
        logging.info('Cumulative Class accuracy:\n%s', np.nanmean(confusion.cumulative_class_acc, axis=1))
        logging.info('Prediction histogram:\n%s',
                     confusion.prediction_histogram[0])

    elif args.action == TEST:
        confusion = ConfusionMatrix(c.n_classes, args.k)
        batch_iter = data.files_batch_test_iter(batch_size=100, num_epochs=1)
        for i, (batch_ids, batch_predictions) in enumerate(classify(
                batch_iter, keep_indices=True)):
            batch_labels = [f.label for f in batch_ids]
            confusion.record(batch_labels, batch_predictions)
            if i % 5 == 4:
                logging.info('Done with batch %d' % i)
        logging.info('Test accuracy:\n%s', confusion.acc)
        logging.info('Cumulative test accuracy:\n%s', confusion.cumulative_acc)
        logging.info('Prediction histogram:\n%s',
                     confusion.prediction_histogram[0])

    elif args.action == EVAL:
        import tensorflow as tf
        import numpy as np
        batch_iter = data.files_batch_test_iter(batch_size=100, num_epochs=1)
        with c._tf_session.as_default() as sess:
            tf_graph = tf.get_default_graph()
            tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            tf_ops = dict((o.name, o) for o in tf_graph.get_operations())
            tf_outs = dict((o.name, o)
                           for op in tf_ops.values()
                           for o in op.outputs)
            tf_vals = dict((v.name, v.eval()) for v in tf_vars)
            run = tf.get_default_session().run
            embed()
