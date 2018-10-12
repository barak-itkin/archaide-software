import numpy as np
import tensorflow as tf
import tensorflow.contrib.image as tf_image


def tf_float32_matrix(obj):
    return tf.cast(tf.convert_to_tensor(obj), dtype=tf.float32)


def float_div(a, b):
    return tf.cast(a, tf.float32) / tf.cast(b, tf.float32)


def round_mult(a, b, dtype=tf.int32):
    return tf.cast(tf.cast(a, tf.float32) * tf.cast(b, tf.float32), dtype)


def tf_translation_matrix(tx, ty):
    return tf_float32_matrix([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])


def tf_rotation_matrix(angle):
    theta = angle / 180 * np.pi
    sin, cos = tf.sin(theta), tf.cos(theta)
    return tf_float32_matrix([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1],
    ])


def tf_chain_matmul(mat, *more_mats):
    result = mat
    for m in more_mats:
        result = tf.matmul(result, m)
    return result


def tf_scale_matrix(sx, sy):
    return tf_float32_matrix([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ])


class TFImageTransform(object):
    def __init__(self, img_shape):
        self.height = img_shape[0]
        self.width = img_shape[1]
        self.src_T_out = tf.eye(3)

    def centered_rotation_matrix(self, angle):
        return tf_chain_matmul(
            tf_translation_matrix(tx=float_div(self.width, 2), ty=float_div(self.height, 2)),
            tf_rotation_matrix(angle),
            tf_translation_matrix(tx=-float_div(self.width, 2), ty=-float_div(self.height, 2))
        )

    def set_size(self, height, width):
        self.height = height
        self.width = width
        return self

    def translate(self, tx, ty):
        self.src_T_out = tf.matmul(
            self.src_T_out,
            tf_translation_matrix(-tx, -ty)
        )
        return self

    def scale(self, sx, sy, resize=True):
        self.src_T_out = tf.matmul(
            self.src_T_out,
            tf_scale_matrix(float_div(1., sx), float_div(1., sy))
        )
        if resize:
            self.set_size(
                height=round_mult(sy, self.height),
                width=round_mult(sx, self.width)
            )
        return self

    def scale_to(self, width, height):
        return self.scale(
            sx=float_div(width, self.width),
            sy=float_div(height, self.height),
        )

    def crop(self, x_start, y_start, width, height):
        self.translate(tx=-x_start, ty=-y_start)
        self.set_size(height=height, width=width)
        return self

    def centered_pad_to_square(self):
        r = tf.maximum(self.width, self.height)
        return self.centered_crop_or_pad(r, r)

    def centered_crop_or_pad(self, width, height):
        dx = (width - self.width) / 2
        dy = (height - self.height) / 2
        return self.crop(x_start=-dx, y_start=-dy,
                         width=width, height=height)

    def random_crop(self, width, height):
        dx = self.width - width
        dy = self.height - height

        x_min, x_max = tf.minimum(0, dx), tf.maximum(0, dx)
        y_min, y_max = tf.minimum(0, dy), tf.maximum(0, dy)

        x_start = tf.cond(tf.equal(x_min, x_max),
                          lambda: x_min,
                          lambda: tf.random_uniform(minval=x_min, maxval=x_max, dtype=tf.int32, shape=()))
        y_start = tf.cond(tf.equal(y_min, y_max),
                          lambda: y_min,
                          lambda: tf.random_uniform(minval=y_min, maxval=y_max, dtype=tf.int32, shape=()))
        # If d > 0 - we need to crop
        # Else we pad
        return self.crop(
            x_start=x_start,
            y_start=y_start,
            width=width, height=height
        )

    def _affine_transform_fit(self, matrix):
        self.src_T_out = tf.matmul(self.src_T_out, matrix)
        pts = tf_float32_matrix([
            [0, 0, 1],
            [0, self.height - 1, 1],
            [self.width - 1, self.height - 1, 1],
            [self.width - 1, 0, 1]
        ])
        pts_trans = tf.transpose(
            tf.matmul(tf.linalg.inv(matrix), tf.transpose(pts))
        )
        pmin = tf.reduce_min(pts_trans, axis=0)
        x_min, y_min = pmin[0], pmin[1]
        pmax = tf.reduce_max(pts_trans, axis=0)
        x_max, y_max = pmax[0], pmax[1]
        new_width = tf.cast(tf.round(x_max - x_min + 1), dtype=tf.int32)
        new_height = tf.cast(tf.round(y_max - y_min + 1), dtype=tf.int32)
        self.crop(
            x_start=x_min,
            y_start=y_min,
            width=new_width,
            height=new_height,
        )
        return self

    def rotate_fit(self, angle):
        return self._affine_transform_fit(
            self.centered_rotation_matrix(-angle)
        )

    def rotate_fit_random(self):
        return self.rotate_fit(tf.random_uniform(minval=0, maxval=360, shape=()))

    def apply(self, img):
        input_shape = tf.shape(img)
        tmp_height = tf.maximum(self.height, input_shape[0])
        tmp_width = tf.maximum(self.width, input_shape[1])
        return tf_image.transform(
            tf.image.pad_to_bounding_box(img, 0, 0, tmp_height, tmp_width),
            tf.reshape(self.src_T_out, (-1,))[:8]
        )[:self.height, :self.width]
