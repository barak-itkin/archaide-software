import numpy as np
import skimage.transform


def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])


def rotation_matrix(angle):
    theta = np.deg2rad(angle)
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([
        [+cos, -sin, 0],
        [+sin, +cos, 0],
        [0, 0, 1],
    ])


def scale_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1],
    ])


class ImageTransform(object):
    def __init__(self, img_shape):
        self.img_shape = list(img_shape[:2])
        self.src_T_out = np.identity(3)

    @property
    def width(self):
        return self.img_shape[1]

    @width.setter
    def width(self, value):
        self.img_shape[1] = value

    @property
    def height(self):
        return self.img_shape[0]

    @height.setter
    def height(self, value):
        self.img_shape[0] = value

    def centered_rotation_matrix(self, angle):
        return (
            translation_matrix(tx=self.width / 2, ty=self.height / 2) @
            rotation_matrix(angle) @
            translation_matrix(tx=-self.width / 2, ty=-self.height / 2)
        )

    def set_size(self, height, width):
        self.height = height
        self.width = width
        return self

    def translate(self, tx, ty):
        self.src_T_out = (
                self.src_T_out @ translation_matrix(-tx, -ty)
        )
        return self

    def scale(self, sx, sy, resize=True):
        self.src_T_out = (
            self.src_T_out @ scale_matrix(1 / sx, 1 / sy)
        )
        if resize:
            self.set_size(height=sy * self.height, width=sx * self.width)
        return self

    def scale_to(self, width, height):
        return self.scale(
            sx=width / self.width,
            sy=height / self.height,
        )

    def crop(self, x_start, y_start, width, height):
        self.translate(tx=-x_start, ty=-y_start)
        self.set_size(height=height, width=width)
        return self

    def centered_pad_to_square(self):
        r = max(self.width, self.height)
        return self.centered_crop_or_pad(r, r)

    def centered_crop_or_pad(self, width, height):
        dx = (width - self.width) / 2
        dy = (height - self.height) / 2
        return self.crop(x_start=-dx, y_start=-dy,
                         width=width, height=height)

    def random_crop(self, width, height):
        dx = self.width - width
        dy = self.height - height
        assert dx >= 0 and dy >= 0
        return self.crop(
            x_start=np.random.randint(low=0, high=dx + 1),
            y_start=np.random.randint(low=0, high=dy + 1),
            width=width, height=height
        )

    def _affine_transform_fit(self, matrix):
        self.src_T_out = self.src_T_out @ matrix
        pts = np.array([
            [0, 0, 1],
            [0, self.height - 1, 1],
            [self.width - 1, self.height - 1, 1],
            [self.width - 1, 0, 1]
        ])
        pts_trans = (np.linalg.inv(matrix) @ pts.T).T
        x_min, y_min = np.min(pts_trans, axis=0)[:2]
        x_max, y_max = np.max(pts_trans, axis=0)[:2]
        new_width = np.round(x_max - x_min + 1)
        new_height = np.round(y_max - y_min + 1)
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
        return self.rotate_fit(np.random.uniform(low=0, high=360))

    def apply(self, img):
        return skimage.transform.warp(
            image=img,
            inverse_map=self.src_T_out,
            output_shape=self.img_shape,
        )
