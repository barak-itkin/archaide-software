import abc
import argparse
import cv2 as cv
import functools
import imageio
import multiprocessing
import numpy as np
import os
import skimage.transform
import skimage.measure
import skimage.color
import traceback

from scipy.ndimage import morphology
from skimage.feature import corner_harris, corner_peaks, corner_subpix

from c3d.classification import ImageDataset
from c3d.util import imgutils


def conditional_cache(cond):
    def decorator(func):
        @functools.wraps(func)
        def result(*args):
            for old_args, old_result in result.cache.items():
                if cond(old_args, args):
                    return old_result
            result.cache[args] = func(*args)
            return result.cache[args]
        result.cache = {}
        return result
    return decorator


def get_labels(mask, part_threshold=0.02):
    """
    Get a labeled mask of connected components within a 2D binary mask. Do so
    while filling holes and dropping parts smaller than a given threshold (in %)
    of the total area.
    """
    assert mask.ndim == 2

    # Fill any existing holes in the mask
    mask_copy = morphology.binary_fill_holes(mask)

    # Label different connected components, with the label 0 serving for the
    # background (deselected) pixels.
    labeled_mask = skimage.measure.label(mask_copy)

    # Count how many components were created
    n_labels = np.max(labeled_mask) + 1

    # Enumerate all components (excluding the background) and remove them from
    # the mask if they are smaller than the given threshold.
    for i in range(1, n_labels):
        part_mask = labeled_mask == i
        if np.sum(part_mask) < part_threshold * mask_copy.size:
            mask_copy[part_mask] = 0

    # Now, re-label after removing the small parts.
    labeled_mask = skimage.measure.label(mask_copy)
    n_labels = np.max(labeled_mask) + 1

    # And return the results
    return labeled_mask, n_labels


def checkerboard_error(img, mask, min_corners=4):
    """
    Compute the "distance" between the given part of the image and a
    checkerboard, using a rough estimation based on Harris corners.

    In a checkerboard pattern, the immediate neighborhood of every corner
    should be invariant to rotation by 180 degrees. So, compute all corners and
    their neighborhoods, and build an error from the geometric mean of all
    differences produced by these rotations.
    """
    if img.ndim == 3:
        gray = skimage.color.rgb2gray(img)
    else:
        gray = np.copy(img)

    # Use a harris corner detector to detect the corners in the image.
    coords = corner_peaks(corner_harris(gray), min_distance=5)
    # Refine corners to subpixel accuracy (and then round) to make sure our
    # neighborhoods are centered on the corner as much as possible.
    # is needed later in the error computation).
    coords = np.round(corner_subpix(gray, coords, window_size=5))
    # Remove NaN corners (introduced when the refinement fails).
    coords = coords[np.sum(np.isnan(coords), axis=1) == 0]
    # Cast back to integer type, to be able to use these as image coordinates.
    coords = coords.astype(np.int)

    # Clip points that are near the edges of the mask, as these are probably
    # false detections.
    small_mask = morphology.binary_erosion(mask, iterations=5)
    good_coords = small_mask[coords[:, 0], coords[:, 1]]

    # Skip patterns that can't detect enough corners. Checkerboard patterns
    # should enable easy detection of at least a few corners, and using this
    # metric with a few corners lets rise to edge cases.
    if np.sum(good_coords) < min_corners:
        return np.inf

    scores = []
    # Take 5x5 neighborhoods around each pixel and subtract these
    for c in coords[good_coords]:
        patch = img[c[0] - 2:c[0] + 3, c[1] - 2:c[1] + 3]
        diff = patch - patch[::-1, ::-1]
        if diff.ndim == 3:
            diff = np.linalg.norm(diff, axis=-1)
        else:
            diff = np.abs(diff)
        scores.append(np.mean(diff))

    # Compute the geometric mean carefully to avoid overflows.
    log_scores = np.log(scores)
    return np.exp(np.mean(log_scores))


def n_channels(img):
    return 1 if img.ndim < 3 else img.shape[-1]


def sample_edges(img, sample_every=1):
    """
    Sample pixels along the edges of the given image.
    """
    nc = n_channels(img)
    return np.vstack((
        img[::sample_every, [0, -1]].reshape((-1, nc)),
        img[[0, -1], ::sample_every].reshape((-1, nc)),
    ))


def filter_outliers(colors):
    gray_level = np.mean(colors, axis=1)
    m = np.median(gray_level)
    s = np.std(gray_level)
    good_mask = np.abs(gray_level - m) < np.maximum(s, 0.3)
    return colors[good_mask]


def img_min_distances(img, colors, norm=np.linalg.norm):
    """
    Compute the minimal color distance from pixels in an image to any color from
    a given list of colors.
    """
    nc = n_channels(img)
    img_pix = img.reshape((-1, 1, nc))
    col_pix = colors.reshape((1, -1, nc))
    d_min = np.min(norm(img_pix - col_pix, axis=-1), axis=-1)
    return d_min.reshape(img.shape[:2])


def grabcut_with_mask_as_rect(img, mask, iter_count=5):
    # Internal grabcut parameters
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Result buffer for grabcut
    gc_mask = np.zeros(img.shape[:2], np.uint8)

    # Grabcut only works with uint8 images
    if 'float' in img.dtype.name:
        img = np.uint8(img * 255)

    cv.grabCut(img, gc_mask, imgutils.mask_bounding_box(mask)._as_tuple(),
               bg_model, fg_model, iter_count, cv.GC_INIT_WITH_RECT)

    # Return identified foreground sections
    return gc_mask == 3


class FgExtractBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _extract(self, img, grabcut=True):
        raise NotImplementedError()

    def extract(self, img, grabcut=True):
        height, width = img.shape[:2]

        # Scale down the image if it's too big, so that we estimate the masks on
        # small images (for significant speedups), and also perform operations with
        # a radius, on a fixed radius.
        if max(height, width) > 500:
            scale = 500 / max(height, width)
            img = skimage.transform.rescale(
                img, scale, mode='reflect'
            )
        else:
            scale = None

        result = self._extract(img, grabcut)
        if np.sum(result) == 0:
            raise RuntimeError('Empty mask produced')

        # Scale back if needed
        if scale is not None:
            result = skimage.transform.resize(
                result, (height, width), mode='reflect'
            ) > 0.5

        return result


class FgExtractWithRuler(FgExtractBase):
    def __init__(self, min_bg_threshold=10./255., threshold_exp_step=1.3):
        self.min_bg_threshold = min_bg_threshold
        self.threshold_exp_step = threshold_exp_step

    def _extract(self, img, grabcut=True):
        # Step 1:
        # Sample pixels along the edges of the image, to obtain reference colors for
        # background colors. Sample only one every 4 pixels as the quality
        # degradation will be minimal, but the computation will be 4x faster.
        edges = sample_edges(img, sample_every=4)
        edges = filter_outliers(edges)

        # Step 2:
        # Compute the minimal distance of each pixel in the image from one of the
        # background colors.
        d_min = img_min_distances(img, edges)

        # Step 3:
        # Compute the threshold that would optimally separate the image into three
        # different parts (ruler, background and foreground).
        @conditional_cache(
            lambda old_args, args: abs(old_args[0] - args[0]) < 1. / 255
        )
        def get_our_labels(th):
            mask = morphology.binary_closing(d_min > th, iterations=3)
            return get_labels(mask)

        # First do an exponential search forward from the given starting threshold,
        # until a separating threshold is found.
        th_low = self.min_bg_threshold
        th_high = th_low * self.threshold_exp_step
        max_th = np.sqrt(n_channels(img))

        while get_our_labels(th_high)[1] < 3 and th_high < max_th:
            th_low, th_high = th_high, min(th_high * self.threshold_exp_step, max_th)

        # Then perform a binary search on the range of the thresholds, to find where
        # exactly we get three parts.
        while th_high - th_low > (1. / 255):
            th = (th_low + th_high) / 2
            if get_our_labels(th)[1] < 3:
                th_low = th
            else:
                th_high = th

        if get_our_labels(th_low)[1] < 3:
            th = th_high
        else:
            th = th_low

        labeled_mask, n_labels = get_our_labels(th)

        if n_labels != 3:
            raise RuntimeError('Failed to deduce optimal threshold')

        # Step 4:
        # Bump the threshold a bit to clean some artifacts.
        for i in range(3):
            if get_our_labels(th + 1. / 255)[1] == 3 and th_high < max_th:
                th += 1. / 255
            else:
                break

        labeled_mask, n_labels = get_our_labels(th)

        # Step 5:
        # Identify which label is the checkerboard ruler and zero it out.
        checkboard_label = min(
            range(1, n_labels),
            key=lambda l: checkerboard_error(img, labeled_mask == l)
        )
        labeled_mask[labeled_mask == checkboard_label] = 0

        result = labeled_mask > 0

        # Do grabcut if requested
        if grabcut:
            result = grabcut_with_mask_as_rect(img, result)
            # Again filter outliers and such
            result = get_labels(result)[0] > 0

        return result


class FgExtractSimple(FgExtractBase):
    def __init__(self, bg_threshold=10./255., margin=2, threshold_exp_step=1.3):
        self.bg_threshold = bg_threshold
        self.margin = margin
        self.threshold_exp_step = threshold_exp_step

    def _extract(self, img, grabcut=True):
        # Step 1:
        # Sample pixels along the edges of the image, to obtain reference colors for
        # background colors. Sample only one every 4 pixels as the quality
        # degradation will be minimal, but the computation will be 4x faster.
        edges = sample_edges(img, sample_every=4)
        edges = filter_outliers(edges)

        # Step 2:
        # Compute the minimal distance of each pixel in the image from one of the
        # background colors.
        d_min = img_min_distances(img, edges)

        th = self.bg_threshold
        while True:
            # Remove small parts and count
            result, n_labels = get_labels(d_min > th)
            if n_labels == 1:
                raise RuntimeError('Failed to deduce threshold')
            result = result > 0
            rect = imgutils.mask_bounding_box(result)

            if (
                rect.xmin <= self.margin or rect.xmin + rect.width >= img.shape[1] - self.margin or
                rect.ymin <= self.margin or rect.ymin + rect.height >= img.shape[0] - self.margin
            ):
                th *= self.threshold_exp_step
            else:
                break

        # Do grabcut if requested
        if grabcut:
            result = grabcut_with_mask_as_rect(img, result)
            # Again filter outliers and such
            result = get_labels(result)[0] > 0

        return result


class CombinedFgExtract(FgExtractBase):
    def __init__(self, *extractors):
        self.extractors = extractors

    def _extract(self, img, grabcut=True):
        for e in self.extractors:
            try:
                mask = e._extract(img, grabcut=True)
                if np.sum(mask) == 0:
                    raise RuntimeError('Empty mask!')
                return mask
            except RuntimeError as ex:
                print('%s failed, with %s' % (type(e), ex))
                continue
        else:
            raise RuntimeError('Failed to extract a mask')


def extract_mask(img, mask):
    result = np.copy(img)
    result[~mask] = 0
    return result


def extract_autocrop_no_bg(img, mask):
    extracted = extract_mask(img, mask)
    rect = imgutils.mask_bounding_box(mask)
    result = np.zeros(
        shape=(rect.height, rect.width, n_channels(img) + 1),
        dtype=img.dtype
    )
    result[:, :, :-1] = extracted[
         rect.ymin:rect.ymin + rect.height,
         rect.xmin:rect.xmin + rect.width,
    ]
    maxval = 1 if 'float' in img.dtype.name else 255
    result[:, :, -1] = maxval * mask[
       rect.ymin:rect.ymin + rect.height,
       rect.xmin:rect.xmin + rect.width,
    ]
    return result


class BaseImageDataset(ImageDataset):
    def file_filter(self, dirpath, filename):
        return '.mask.' not in filename and '.fg.' not in filename


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


class BatchExtract(object):
    def __init__(self, store_mask, store_extracted, overwrite, base_dir=None, dst_dir=None):
        self.store_mask = store_mask
        self.store_extracted = store_extracted
        self.overwrite = overwrite
        self.extractor = CombinedFgExtract(
            FgExtractWithRuler(), FgExtractSimple()
        )
        self.base_dir = expand_path(base_dir)
        self.dst_dir = expand_path(dst_dir)

    def _work_needed(self, path):
        return self.overwrite or not os.path.exists(path)

    def process_file(self, path):
        try:
            path = expand_path(path)
            print('Processing %s' % path)
            if self.dst_dir and self.base_dir:
                dst_base = os.path.join(self.dst_dir, os.path.relpath(path, self.base_dir))
            else:
                dst_base = path
            print('Will be saved in %s' % dst_base)

            mask_path = dst_base + '.mask.png'
            fg_path = dst_base + '.fg.png'

            if ((self.store_mask and self._work_needed(mask_path)) and
                    not (self.store_extracted and self._work_needed(fg_path))):
                return

            dst_dir = os.path.dirname(dst_base)
            print('Under %s' % dst_dir)
            os.makedirs(dst_dir, exist_ok=True)

            img = imageio.imread(path)
            mask = self.extractor.extract(img)

            if self.store_mask and self._work_needed(mask_path):
                imageio.imwrite(mask_path, np.uint8(mask * 255))

            if self.store_extracted and self._work_needed(fg_path):
                imageio.imwrite(fg_path, extract_autocrop_no_bg(img, mask))

            return 1
        except RuntimeError as e:
            print('Exception with %s' % path)
            traceback.print_exc()
            return 0
        except Exception as e:
            print('Unknown exception with %s' % path)
            traceback.print_exc()
            return 0


def main():
    parser = argparse.ArgumentParser(
        description='Extract the foreground of a given sherd'
    )
    parser.add_argument('input_path', type=str,
                        help='Input path (file or directory)')
    parser.add_argument('--alternate_base', default=None, type=str,
                        help='Specify a different base directory for '
                        'saved images (with the same directory structure as '
                        'the input folder)')
    parser.add_argument('--save_fg', help='Save the extracted FG image',
                        default=False, action='store_true')
    parser.add_argument('--save_mask', help='Store the extracted mask',
                        default=False, action='store_true')
    parser.add_argument('--num_workers', help='Number of workers',
                        default=1, type=int)
    parser.add_argument('--overwrite', help='Overwrite existing masks',
                        default=False, action='store_true')

    args = parser.parse_args()

    if not args.save_mask and not args.save_fg:
        print('Must specify at least one result to save')
        exit(1)

    if not os.path.exists(args.input_path):
        print('Input path not found')
        exit(1)

    if os.path.isdir(args.input_path):
        base_dir = args.input_path
        data = BaseImageDataset(base_dir)
        all_paths = set(
            data.file_path(file_id)
            for file_id in data.file_ids_iter()
        )
    else:  # Path to a single file
        base_dir = os.path.dirname(args.input_path)
        all_paths = {args.input_path}

    be = BatchExtract(store_extracted=args.save_fg, store_mask=args.save_mask,
                      overwrite=args.overwrite, base_dir=base_dir,
                      dst_dir=args.alternate_base)

    if args.num_workers <= 1:
        success = list(map(be.process_file, all_paths))
    else:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            success = pool.map(be.process_file, all_paths)

    print('Success rate: %.2f%%' % (100 * np.mean(success)))


if __name__ == '__main__':
    main()
