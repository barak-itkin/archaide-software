import argparse
import logging
import numpy as np
import os
import pickle
import sys

from c3d.classification import ConfusionMatrix, ImageDataset
from .model import Classifier, box_crop

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(stream=sys.stdout, format=FORMAT, level=logging.DEBUG)

TRAIN = 'train'
EVAL = 'eval'
CLASSIFY = 'classify'
ACTIONS = [TRAIN, EVAL, CLASSIFY]


def make_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Appearance based classifier')
    parser.add_argument('action', type=str, choices=ACTIONS,
                        help="The action to take on the data (`train' the "
                        "classifier, `eval'uate the training process, or "
                        "`classify' given inputs)")
    parser.add_argument('data_dir', type=str, metavar='data-dir',
                        help='Path to the directory containing the image data')
    parser.add_argument('--model-path', type=str, default='appearance.model',
                        required=False,
                        help='Path used to load/save the classifier (model) '
                        'when training/classifying.')
    parser.add_argument('--resnet-dir', type=str,
                        default='tensorflow-resnet-pretrained-20160509',
                        required=False,
                        help='The directory containing the pre-trained ResNet '
                        'models')
    parser.add_argument('--enable-tensorflow-logs', default=False,
                        action='store_true',
                        help='Show logging by TensorFlow (noisy, discouraged)')

    train_args = parser.add_argument_group('train arguments')
    train_args.add_argument('--data-feature-cache', type=str, default=None,
                            required=False,
                            help='DEVELOPMENT ONLY: If specified, cache the '
                            'features of dataset images to this path, to allow '
                            'future trainings to be quicker. You must clear '
                            'this when updating the dataset.')

    eval_args = parser.add_argument_group('evaluation arguments')
    eval_args.add_argument('--train-to-test-ratio', type=int, default=6,
                           required=False,
                           help='The ratio (num of train samples) / (num of '
                           'test samples)')
    eval_args.add_argument('--num-runs', type=int, default=10, required=False,
                           help='The number of times to train the model in the '
                           'evaluation process')

    train_args = parser.add_argument_group('classification arguments')
    train_args.add_argument('--k', type=int, default='1', required=False,
                            help='Number of best classes to predict')
    return parser


args = make_parser().parse_args()
will_train = args.action in [TRAIN, EVAL]
train_test_ratio = args.train_to_test_ratio if args.action == EVAL else None

if not args.enable_tensorflow_logs:
    import tensorflow
    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
    # And also silence the native code
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Stage #1: construct the dataset we'll work with
data = ImageDataset(args.data_dir, train_test_ratio)

# If we are training:
if will_train:
    # Stage #2: Construct the classifier object
    c = Classifier(data, args.resnet_dir, cache_path=args.data_feature_cache)
    # Stage #3: pre-process the inputs
    c.cache_features()


def classify(files_batch_iter, keep_indices=False):
    for batch_ids, batchs_imgs in files_batch_iter:
        for file_id, img in zip(batch_ids, batchs_imgs):
            prediction_indices = c.predict_top_k([box_crop(img)], args.k)[0]
            if keep_indices:
                yield file_id, prediction_indices
            else:
                predictions = [c.index_to_label[i] for i in prediction_indices]
                yield file_id, predictions


# And now diverge by action
if args.action == TRAIN:
    c.train()
    with open(args.model_path, 'wb') as f:
        pickle.dump(c, f)

elif args.action == CLASSIFY:
    with open(args.model_path, 'rb') as f:
        c = pickle.load(f)
    c.dataset = data
    c.cache_path = args.data_feature_cache

    for file_id, predictions in classify(
            data.files_batch_iter(batch_size=1, num_epochs=1)):
        # Intentionally print and don't log, so that the format is easy to
        # expect and parse.
        print(data.file_path(file_id))
        for i, p in enumerate(predictions):
            print('%3d: %s' % (i + 1, p))

elif args.action == EVAL:
    accs_train = []
    accs_test = []
    for i in range(args.num_runs):
        logging.info('****** Run #%d (of %d) ******', (i + 1), args.num_runs)
        c.dataset.shuffle()
        c.train()
        confusion_train = ConfusionMatrix(len(c.label_to_index), args.k)
        for file_id, predictions in classify(
                data.files_batch_train_iter(batch_size=1, num_epochs=1),
                keep_indices=True):
            label_index = c.label_to_index[file_id.label]
            confusion_train.record([label_index], predictions)
        accs_train.append(confusion_train.cumulative_acc)
        logging.info('Train accuracy (by top-K): %s', accs_train[-1])

        confusion_test = ConfusionMatrix(len(c.label_to_index), args.k)
        for file_id, predictions in classify(
                data.files_batch_test_iter(batch_size=1, num_epochs=1),
                keep_indices=True):
            label_index = c.label_to_index[file_id.label]
            confusion_test.record([label_index], predictions)
        accs_test.append(confusion_test.cumulative_acc)
        logging.info('Test accuracy (by top-K): %s', accs_test[-1])

    logging.info('****** Overall ******')
    logging.info('Train accuracy (by top-K): %s', np.mean(accs_train, 0))
    logging.info('Test accuracy (by top-K): %s', np.mean(accs_test, 0))
else:
    raise NotImplementedError('Action "%s" is not supported!' % args.action)
