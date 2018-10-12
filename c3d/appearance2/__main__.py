import argparse
import logging
import os
import pickle
import sys

from c3d.classification import ConfusionMatrix, ImageDataset
from c3d.util import imgutils
from .model import Classifier
from .input import RandomTransformCropper, fill_bg_with_black


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(stream=sys.stdout, format=FORMAT, level=logging.INFO)

TRAIN = 'train'
TEST = 'test'
CLASSIFY = 'classify'
ACTIONS = [TRAIN, TEST, CLASSIFY]


def make_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Appearance based classifier')

    parser.add_argument('model_path', type=str, metavar='model-path',
                        help='Path for saving/loading the model file')
    parser.add_argument('data_dir', type=str, metavar='data-dir',
                        help='Path to folder containing the data')

    parser.add_argument('--disable-tensorflow-logs', default=False,
                        action='store_true', help='Hide logging by TensorFlow')
    parser.add_argument('--k', type=int, default=3, required=False,
                        help='Number of top classes to predict')
    parser.add_argument('--train-to-test-ratio', type=int, default=6,
                        required=False,
                        help='The ratio (num of train samples) / (num of '
                        'test samples)')

    actions = parser.add_subparsers(dest='action')

    train_parser = actions.add_parser(TRAIN, help='Train the classifier')
    train_parser.add_argument('--cache_dir', type=str,
                              default='cache', required=False,
                              help="Directory for caching features during the "
                              "train (in case it's interrupted)")
    train_parser.add_argument('--summary_dir', type=str,
                              default='summary', required=False,
                              help="Directory for writing tensorboard summaries")
    train_parser.add_argument('--resnet-dir', type=str, default=None,
                              required=False,
                              help='The directory containing the pre-trained ResNet '
                              'models')

    test_parser = actions.add_parser(TEST,
                                     help='Evaluate the training process. Use '
                                     'ONLY with the same data folder as for '
                                     'the training (and the same '
                                     '--train-to-test-ratio)')

    classify_parser = actions.add_parser(CLASSIFY,
                                         help='Classify given inputs')

    return parser


def assert_or_quit(condition, msg):
    if not condition:
        print(msg)
        exit(1)


args = make_parser().parse_args()
assert_or_quit(args.action, 'No action specified')

train_test_ratio = args.train_to_test_ratio if args.action != CLASSIFY else None

if args.disable_tensorflow_logs:
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    # And also silence the native code
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = ImageDataset(args.data_dir, train_to_test_ratio=train_test_ratio)
data.balance = False
transformer = RandomTransformCropper(size=224)

data.per_run_filters.extend([
    transformer.apply,
    imgutils.img_to_float,
    fill_bg_with_black
])

if args.action == TRAIN:
    transformer.random_transform = True
    assert_or_quit(args.resnet_dir,
                   '--resnet-dir must be specified for training a model!')
    c = Classifier(
        data, args.resnet_dir,
        cache_dir=args.cache_dir,
        summary_dir=args.summary_dir
    )
    try:
        c.train()
    except (KeyboardInterrupt, InterruptedError) as e:
        logging.warning('Training interrupted, saving our current model')
    with open(args.model_path, 'wb') as f:
        pickle.dump(c, f)
else:
    transformer.random_transform = False
    with open(args.model_path, 'rb') as f:
        c = pickle.load(f)
    c.dataset = data
    if args.resnet_dir:
        c.resnet_dir = args.resnet_dir

    def classify(files_batch_iter, keep_indices=False):
        for batch_ids, batchs_imgs in files_batch_iter:
            batch_prediction_indices = c.predict_top_k(batchs_imgs, args.k)
            if keep_indices:
                yield batch_ids, batch_prediction_indices
            else:
                yield batch_ids, [[c.index_to_label[i] for i in prediction_indices]
                                  for prediction_indices in batch_prediction_indices]

    if args.action == CLASSIFY:
        for batch_ids, batch_predictions in classify(
                data.files_batch_iter(batch_size=c.batch_size, num_epochs=1)):
            for file_id, predictions in zip(batch_ids, batch_predictions):
                # Intentionally print and don't log, so that the format is easy to
                # expect and parse.
                print(data.file_path(file_id))
                for i, p in enumerate(predictions):
                    print('%3d: %s' % (i + 1, p))

    elif args.action == TEST:
        confusion = ConfusionMatrix(c.n_classes, args.k)
        batch_iter = data.files_batch_test_iter(batch_size=c.batch_size, num_epochs=1)
        try:
            for i, (batch_ids, batch_predictions) in enumerate(classify(
                    batch_iter, keep_indices=True)):
                batch_labels = [c.label_to_index[f.label] for f in batch_ids]
                confusion.record(batch_labels, batch_predictions)
                if i % 5 == 4:
                    logging.info('Done with batch %d' % i)
        except (KeyboardInterrupt, InterruptedError) as e:
            logging.warning('Testing interrupted, will report results so far')
        logging.info('Test sample count:\n%s', confusion.n)
        logging.info('Test accuracy:\n%s', confusion.acc)
        logging.info('Cumulative test accuracy:\n%s', confusion.cumulative_acc)
        logging.info('Input histogram:\n%s',
                     confusion.input_histogram)
        logging.info('Prediction histogram:\n%s',
                     confusion.prediction_histogram[0])
