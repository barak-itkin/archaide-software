import numpy as np
import skimage.transform
import tensorflow as tf
from c3d.util.rect import CvRect


def scale(img, size):
    """Scale a square image to the given size (width and height)."""
    return skimage.transform.resize(img, (size, size))


def box_crop(img, size=None):
    """Crop an image (width X height [X channels]) to a square, and scale if size is specified."""
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    if size is None:
        return crop_img
    else:
        return scale(crop_img, size)


def box_fit(img, size=224):
    """Symmetric pad an image (width X height [X channels]) to a square, and scale if size is
    specified."""
    if img.shape[0] > img.shape[1]:
        pad_img = img[:, symmetric_pad(0, img.shape[1], img.shape[0])]
    else:
        pad_img = img[symmetric_pad(0, img.shape[0], img.shape[1]), :]
    if size is None:
        return pad_img
    else:
        return scale(pad_img, size)


def symmetric_pad(start, end, width):
    """Return an array of [s, ..., s, s + 1, ..., e - 1, e, ..., e]."""
    size = end - start
    if size > width:
        raise NotImplementedError('Only supports padding, not cropping')
    elif size == width:
        return np.arange(start, end)
    else:
        pad = width - size
        pad_s = pad // 2
        pad_e = pad - pad_s
        result = np.zeros(width, dtype=np.int32) + start
        result[-pad_e:] = end - 1
        result[pad_s:(pad_s+size)] = np.arange(start, end)
        return result


def mask_bounding_box(mask, padding=0):
    mask_xs = np.arange(mask.shape[1])[np.any(mask, axis=0)]
    mask_ys = np.arange(mask.shape[0])[np.any(mask, axis=1)]
    return CvRect(
        xmin=mask_xs[0] - padding,
        ymin=mask_ys[0] - padding,
        width=mask_xs[-1] - mask_xs[0] + 1 + 2 * padding,
        height=mask_ys[-1] - mask_ys[0] + 1 + 2 * padding
    )


def tf_mask_bounding_box(mask, padding=0):
    if mask.dtype != tf.bool:
        raise ValueError('Expected boolean mask!')
    coords = tf.where(mask)
    c_min = tf.reduce_min(coords, axis=0)
    c_max = tf.reduce_max(coords, axis=0)
    return CvRect(
        xmin=c_min[1] - padding,
        ymin=c_min[0] - padding,
        width=c_max[1] - c_min[1] + 1 + 2 * padding,
        height=c_max[0] - c_min[0] + 1 + 2 * padding
    )


def autocrop(img, mask=None):
    if img.ndim != 3:
        raise ValueError('Expected an image with color channels, but input '
                         'dimension is not 3!')
    if mask is None:
        if img.shape[2] != 4:
            raise ValueError('Expected RGBA image if a mask is not '
                             'provided!')
        mask = img[:, :, 3] > 0
    box = mask_bounding_box(mask)
    return img[
       box.ymin:box.ymax + 1,
       box.xmin:box.xmax + 1,
    ]


def tf_autocrop(img):
    mask = img[:, :, 3] > 0
    box = tf_mask_bounding_box(mask)
    return img[
       box.ymin:box.ymax + 1,
       box.xmin:box.xmax + 1,
    ]


def img_to_uint8(img):
    if type(img) is not np.ndarray:
        raise ValueError('Images should be Numpy ndarray objects!')
    elif 'float' in img.dtype.name and img.max() <= 1:
        return np.uint8(255 * img)
    else:
        return img


def img_to_float(img):
    if type(img) is not np.ndarray:
        raise ValueError('Images should be Numpy ndarray objects!')
    elif 'float' not in img.dtype.name and img.max() <= 255:
        return np.float32(img / 255.)
    else:
        return img


def load_resnet_mean_bgr():
    # TODO: Obtain the mean RGB values of ResNet.
    # For now we're using a value that was picked by manual checks on data
    return 20. / 255.


def resnet_preprocess(img_or_batch):
    """Prepare a uint8 RGB image for ResNet input"""
    img_or_batch = np.asanyarray(img_or_batch)
    if not 3 <= img_or_batch.ndim <= 4 or img_or_batch.shape[-1] != 3:
        raise ValueError('Expected RGB image/batch, but got shape=%s' % str(img_or_batch.shape))
    img_or_batch = img_to_float(img_or_batch)
    bgr = img_or_batch[..., [2, 1, 0]] - load_resnet_mean_bgr()
    return bgr
