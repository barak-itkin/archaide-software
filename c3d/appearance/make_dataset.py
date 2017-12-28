import argparse
import functools
import multiprocessing
import numpy as np
import os
import PIL.Image

from c3d.classification import dataset
from c3d.util import imgutils


def make_parser():
    parser = argparse.ArgumentParser(
        description='Create an appearance dataset from a directory of source images')
    parser.add_argument('input_dir', type=str, metavar='data-dir',
                        help='Path to folder containing the source data')
    parser.add_argument('output_dir', type=str, metavar='out-dir',
                        help='Path to folder for writing the augmented dataset')
    parser.add_argument('--num_jobs', type=int, metavar='num-jobs', default=1,
                        help='Number of jobs to run in parallel')
    return parser


def run(input_dir, output_dir, total_workers=None, worker_id=None):
    raw_data = dataset.ImageDataset(input_dir)
    if total_workers is not None and total_workers > 1:
        raw_data.num_sherds = total_workers
        raw_data.sherd_id = worker_id
    augmented_data = dataset.ImageDataset(output_dir)
    for f_id, img in raw_data.files_iter():
        # Take the image type from the file name extension, otherwise fall-back
        # to PNG (this should not happen, as the ImageDataset will only chose
        # images based on their file extension).
        f_name = os.path.basename(raw_data.file_name(f_id))
        save_type = f_name.rsplit('.', 1)[-1].lower() if '.' in f_name else 'png'
        if save_type == 'jpg':
            save_type = 'jpeg'
        for ns in augment_image(img):
            new_id = dataset.make_augmented_id(f_id, ns.name)
            new_path = augmented_data.file_path(new_id)
            new_dir = os.path.dirname(new_path)
            # Since we are multi-processing, there's no point in checking for directory
            # existence as another job may create it. Instead, just specify no errors
            # should be raised if the folder exists
            os.makedirs(new_dir, exist_ok=True)
            pil_img = PIL.Image.fromarray(np.uint8(255 * ns.sample))
            pil_img.save(new_path, save_type)


class Worker:
    def __init__(self, input_dir, output_dir, num_workers):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_workers = num_workers

    def work(self, id):
        return run(input_dir=self.input_dir, output_dir=self.output_dir, total_workers=self.num_workers, worker_id=id)


def main():
    args = make_parser().parse_args()
    if args.num_jobs > 1:
        worker = Worker(args.input_dir, args.output_dir, args.num_jobs)
        pool = multiprocessing.Pool(args.num_jobs)
        pool.map(worker.work, range(args.num_jobs))
    else:
        run(input_dir=args.input_dir, output_dir=args.output_dir)


def img_scale(img):
    yield dataset.NamedSample(
        sample=imgutils.box_fit(img, 224),
        name='box_fit_224')
    yield dataset.NamedSample(
        sample=imgutils.box_fit(img, 284)[30:254, 30:254, :],
        name='box_fit_284')
    yield dataset.NamedSample(
        sample=imgutils.box_fit(img, 344)[60:284, 60:284, :],
        name='box_fit_344')
    yield dataset.NamedSample(
        sample=imgutils.box_crop(img, 224),
        name='box_crop_224')


def img_flip(img):
    yield dataset.NamedSample(
        sample=img,
        name=None)
    yield dataset.NamedSample(
        sample=np.flip(img, 0),
        name='hflip')
    yield dataset.NamedSample(
        sample=np.flip(img, 1),
        name='vflip')


def augment_image(img):
    named_images = [dataset.NamedSample(name=None, sample=img)]
    named_images = dataset.augment(named_images, img_scale)
    named_images = dataset.augment(named_images, img_flip)
    return named_images


def prepare_img(img):
    if type(img) is not np.ndarray:
        raise ValueError('Images should be Numpy ndarray objects!')
    if img.ndim != 3:
        raise ValueError('Images should be RGB (3 dimensional arrays!)')
    if not img.shape[0] == img.shape[1] == 224:
        img = imgutils.box_crop(img, 224)
    if 'float' in img.dtype.name and img.max() <= 1:
        img = np.uint8(255 * img)
    return img


if __name__ == '__main__':
    main()
