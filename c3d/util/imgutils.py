import numpy as np
import skimage.transform


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


def load_resnet_mean_bgr():
    raise NotImplementedError()


def resnet_preprocess(img):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    mean_bgr = load_resnet_mean_bgr()
    out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out -= mean_bgr
    return out
