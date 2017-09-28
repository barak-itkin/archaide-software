import c3d.datamodel
import numpy


def outline_from_fracture(fracture):
    up = numpy.array([0, 0, 1])
    side = numpy.cross(up, fracture.normal)
    points = [
        [numpy.dot(side, v), v[2]]
        for v in fracture.vertices
    ]
    return c3d.datamodel.Outline(points)