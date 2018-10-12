from c3d.util.imgtransform import ImageTransform
from c3d.util.imgutils import autocrop
import numpy as np
import tensorflow as tf

import c3d.util.tfimgtransform as tfimgtransform
from c3d.util.imgutils import tf_autocrop, img_to_float


def fill_bg_with_black(img):
    assert 'float' in img.dtype.name
    if img.shape[2] == 3:  # RGB (not RGBA)
        return img
    elif img.shape[2] == 4:  # RGBA
        #      RGB   times    ALPHA
        return img[..., :3] * img[..., [3]]
    else:
        raise ValueError('Image should be RGB or RGBA!')


class RandomTransformCropper(object):
    def __init__(self, size, scale_min=0.5, scale_max=1.5):
        # The scale parameters are ratios relative to the original size
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.autocrop = False

    def _random_size(self):
        # TODO - Maybe distribute better (more small, less large)
        return self.size * np.random.uniform(
            low=self.scale_min,
            high=self.scale_max
        )

    def apply(self, img):
        if self.autocrop:
            img = autocrop(img)
        # Create the image transformation
        t = ImageTransform(img.shape)
        # Rotate in a random angle
        t = t.rotate_fit_random()
        # Pad to a square (which we need for scaling)
        t = t.centered_pad_to_square()
        # Apply random scale-crop/pad
        new_size = self._random_size()
        t = t.scale_to(new_size, new_size)
        # Crop or pad as needed
        if new_size >= self.size:
            t.random_crop(self.size, self.size)
        else:
            t.centered_crop_or_pad(self.size, self.size)
        # Finally, apply to obtain the final image
        return t.apply(img)


def prepare_img(img, size=224):
    img = img_to_float(img)
    img = fill_bg_with_black(img)
    t = ImageTransform(img.shape)
    t = t.centered_pad_to_square()
    t = t.scale_to(size, size)
    return t.apply(img)


def make_train_dataset(py_dataset, label_to_index, batch_size):
    train_files = []
    train_file_labels = []

    for fid in py_dataset.file_ids_train_iter():
        train_files.append(py_dataset.file_path(fid))
        train_file_labels.append(label_to_index[fid.label])

    def _read_img(filename):
        im_str = tf.read_file(filename)
        # Decode has a range of 0-255, and returns floating images
        image_decoded = tf.image.decode_png(im_str) / 255
        return image_decoded

    def _rotate_img(img):
        T = tfimgtransform.TFImageTransform(tf.shape(img))
        return T.rotate_fit_random().apply(img)

    def _make_square(img):
        T = tfimgtransform.TFImageTransform(tf.shape(img))
        T = T.centered_pad_to_square()
        new_size = tf.random_uniform(minval=224/2, maxval=224*2, shape=())
        T = T.scale_to(new_size, new_size)
        return T.random_crop(224, 224).apply(img)

    def _autocrop(img):
        return tf_autocrop(img)

    def _remove_bg(img):
        color = img[..., :3]
        mask = tf.expand_dims(img[..., 3], -1)
        result = color * tf.cast(mask > 0, tf.float32) # Fade to black
        return result

    def _color_augment(img):
        # Luminance shift
        lum_mult = tf.random_uniform(
            minval=0.8, maxval=1.2, shape=(1, 1, 1)
        )
        # Color balance shift
        color_mult = tf.random_uniform(
            minval=0.9, maxval=1.1, shape=(1, 1, 3)
        )
        # Recolor by both shifts
        return tf.minimum(
            1.,
            img * (lum_mult * color_mult)
        )

    def _prep_img(path):
        img = _read_img(path)
        img = _rotate_img(img)
        img = _autocrop(img)
        img = _make_square(img)
        img = _remove_bg(img)
        img = _color_augment(img)
        return img

    dataset = tf.data.Dataset.from_tensor_slices((train_files, train_file_labels))
    dataset = dataset.shuffle(len(train_files))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda p, label: (_prep_img(p), label), num_parallel_calls=10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    dataset_iter = dataset.make_one_shot_iterator()
    dataset_next = dataset_iter.get_next()
    images_next = dataset_next[0]
    labels_next = dataset_next[1]
    return images_next, labels_next
