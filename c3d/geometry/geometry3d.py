# About directions:
#   POINTS are always COLUMN vectors
#   DIRECTIONS are always ROW vectors

import numpy, numpy.linalg


def normalize(vec):
    norm = numpy.linalg.norm(vec)
    return vec / norm


def project(src, onto, onto_normalized=False):
    if not onto_normalized:
        onto = normalize(onto)
    return numpy.dot(src, onto) * onto


def remove_component(vec, component, component_normalized=False):
    return vec - project(vec, component, component_normalized)


def row(vec, copy=False):
    vec = numpy.array(copy) if copy else vec
    vec.shape = (1, vec.size)
    return vec


def column(vec, copy=False):
    vec = numpy.array(copy) if copy else vec
    vec.shape = (vec.size, 1)
    return vec


def normalize_by_ccordinate_z(points):
    if points.ndim < 2:
        points = column(points, copy=True)
    z_vals = points[2, :]
    # Make a row vector from the Z values
    z_vals.shape = (1, points.shape[1])
    # And repeat it twice so that it spans both X and Y rows
    repz = z_vals.repeat(2, axis=0)
    # Now divide the X and Y by the Z values
    points[0:2] = numpy.divide(points[0:2], repz)


class Camera:
    def __init__(self, eye, direction, up, f=1):
        self.eye = numpy.array(eye) if eye else None
        self.direction = normalize(numpy.array(direction))
        self.up = normalize(remove_component(numpy.array(up), self.direction))
        # Numpy cross product is actually giving the opposite sign from the right hand rule
        self.right = -normalize(numpy.cross(self.direction, self.up))

        # The eye should be a column vector
        self.eye = column(self.eye) if eye else None
        # The directions should be row vectors
        self.direction = row(self.direction)
        self.up = row(self.up)
        self.right = row(self.right)

        self.f = f

    def get_matrix_3x3(self):
        return numpy.concatenate((
            self.right,
            self.up,
            self.direction,
        ))

    # Each point should be a column vector
    def project_perspective(self, points):
        if self.eye is None:
            raise ValueError("Can't do perspective transform of eye-less camera!")
        if points.ndim < 2:
            points = column(points, copy=True)
        # Repeat the eye column to span all point columns
        repeye = self.eye.repeat(points.shape[1], axis=1)
        result = numpy.dot(self.get_matrix_3x3(), (points - repeye))
        normalize_by_ccordinate_z(result)
        return result * self.f

    # Each point should be a column vector
    def project_orthographic(self, points):
        if points.ndim < 2:
            points = column(points, copy=True)
        # Repeat the eye column to span all point columns
        if self.eye is None:
            repeye = self.eye.repeat(points.shape[1], axis=1)
            points = points - repeye
        result = numpy.dot(self.get_matrix_3x3(), points)
        return result
