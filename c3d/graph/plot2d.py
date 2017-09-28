from abc import ABCMeta
from collections import namedtuple
from functools import lru_cache
import math
from xml.etree import ElementTree

from c3d.geometry import geometry2d
from c3d.geometry.geometry2d import CX, CY


class Color(namedtuple('Color', ['r', 'g', 'b'])):
    def hex(self):
        return '#%02x%02x%02x' % tuple(int(c * 255) for c in self)


class Colors:
    BLACK = Color(0., 0., 0.)
    WHITE = Color(1., 1., 1.)
    RED = Color(0.8, 0.05, 0.1)
    GREEN = Color(0.05, 0.8, 0.1)


class StyleContext:
    def __init__(self, **kwargs):
        self.properties = dict(kwargs)
        self.set_if_unset('fill', None)
        self.set_if_unset('stroke', Colors.BLACK)
        self.set_if_unset('stroke_width', 2)

    def clone(self):
        return StyleContext(**self.properties)

    def get_or(self, prop, val, action=lambda x: x):
        if prop in self.properties:
            return action(self.properties[prop])
        else:
            return val

    @staticmethod
    def color2str(color):
        return color.hex() if color else 'none'

    def set_if_unset(self, prop, val):
        self.properties[prop] = self.get_or(prop, val)

    def fork_update(self, **kwargs):
        new_args = dict(self.properties)
        for key, val in kwargs.items():
            new_args[key] = val
        return StyleContext(**new_args)

    def get_svg_attrs(self):
        attrs = {}

        attrs['fill'] = self.get_or('fill', 'none', StyleContext.color2str)
        attrs['stroke'] = self.get_or('stroke', 'none', StyleContext.color2str)
        if attrs['stroke'] != 'none' and 'stroke_width' in self.properties:
            attrs['stroke-width'] = str(self.properties['stroke_width'])

        return attrs


class Shape(metaclass=ABCMeta):
    def __init__(self):
        self.style_context = None
        self.box_cache = None

    def get_svg_attrs(self):
        raise NotImplementedError()

    def get_svg_name(self):
        raise NotImplementedError()

    def to_svg(self):
        element = ElementTree.Element(self.get_svg_name(), self.get_svg_attrs())
        if self.style_context:
            for name, value in self.style_context.get_svg_attrs().items():
                element.set(name, value)
        return element

    def _get_points(self):
        raise NotImplementedError()

    def transform(self, transform):
        self.box_cache = None

    def compute_box(self):
        self.box_cache = geometry2d.BoundingBox.from_points(self._get_points())
        return self.box_cache

    @property
    def box(self):
        return self.box_cache or self.compute_box()


class Line(Shape):
    def __init__(self, src, dst):
        super.init()
        self.src = src
        self.dst = dst

    def get_svg_attrs(self):
        return {
            'x1': '%f' % self.src[CX],
            'y1': '%f' % self.src[CY],
            'x2': '%f' % self.dst[CX],
            'y2': '%f' % self.dst[CY],
        }

    def transform(self, transform):
        self.src = transform(self.src)
        self.dst = transform(self.dst)
        super().transform(transform)

    def get_svg_name(self):
        return 'line'

    def _get_points(self):
        return [self.src, self.dst]


class Polygon(Shape):
    def __init__(self, points):
        super().__init__()
        self.points = list(points)

    def get_svg_attrs(self):
        return {
            'points': ', '.join('%f %f' % (p[CX], p[CY]) for p in self.points),
        }

    def get_svg_name(self):
        return 'polygon'

    def transform(self, transform):
        self.points = [transform(point) for point in self.points]
        super().transform(transform)

    def _get_points(self):
        return self.points


def euclid_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


class Circle(Shape):
    def __init__(self, cx, cy, r):
        super().__init__()
        self.cx = cx
        self.cy = cy
        self.r = r

    def get_svg_attrs(self):
        return {
            'cx': str(self.cx),
            'cy': str(self.cy),
            'r': str(self.r),
        }

    def get_svg_name(self):
        return 'circle'

    def transform(self, transform):
        fake_points = [
            (self.cx - self.r, self.cy),
            (self.cx + self.r, self.cy),
            (self.cx, self.cy - self.r),
            (self.cx, self.cy + self.r),
        ]
        new_points = [transform(point) for point in fake_points]
        self.cx = sum(p[0] for p in new_points)
        self.cx = sum(p[1] for p in new_points)
        self.r = sum(euclid_distance((self.cx, self.cy), p) for p in new_points) / len(new_points)

    def compute_box(self):
        self.box_cache = geometry2d.BoundingBox.from_points([
            (self.cx - self.r, self.cy - self.r),
            (self.cx + self.r, self.cy + self.r),
        ])
        return self.box_cache

    def _get_points(self):
        return NotImplementedError()


class Polyline(Shape):
    def __init__(self, points):
        super().__init__()
        self.points = list(points)

    def get_svg_attrs(self):
        return {
            'points': ', '.join('%f %f' % (p[CX], p[CY]) for p in self.points),
        }

    def get_svg_name(self):
        return 'polyline'

    def transform(self, transform):
        self.points = [transform(point) for point in self.points]
        super().transform(transform)

    def _get_points(self):
        return self.points


class Graph:
    def __init__(self):
        self.bounding_box = None
        self.style_context = StyleContext()
        self.shapes = []

    def update_style(self, **kwargs):
        self.style_context = self.style_context.fork_update(**kwargs)

    def add_shape(self, kind, *args, **kwargs):
        shape = kind(*args, **kwargs)
        shape.style_context = self.style_context
        self.shapes.append(shape)
        return shape

    def transform(self, transform):
        for shape in self.shapes:
            shape.transform(transform)

    def autoscale(self):
        if self.shapes:
            self.bounding_box = geometry2d.BoundingBox.combine(
                shape.box for shape in self.shapes
            )

    def export_svg(self, width=None, height=None, as_string=True, encoding='utf-8'):
        root = ElementTree.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'viewBox': '%f %f %f %f' % (
                self.bounding_box.min_x, self.bounding_box.min_y,
                self.bounding_box.width, self.bounding_box.height,
            ),
            'preserveAspectRatio': 'xMidYMid meet'
        })

        if width is not None:
            root.set('width', str(width))
        if height is not None:
            root.set('height', str(width))

        for shape in self.shapes:
            root.append(shape.to_svg())

        if as_string:
            return ElementTree.tostring(root, encoding).decode(encoding)
        else:
            return root
