from collections import namedtuple

Point2D = namedtuple('Point2D', 'x y')


# Prefer accessing X/Y values via indexing, to allow much more input types such as numpy
# arrays, tuples, vectors, etc.
CX = 0
CY = 1


# Accepted numerical error when matching a condition.
# Currently 0, will adjust later if needed.
EPS = 0


def in_range(val, min, max):
    return min - EPS <= val <= max + EPS


class BoundingBox(namedtuple('BoundingBox', 'min_x min_y max_x max_y')):
    @property
    def min_point(self):
        return Point2D(self.min_x, self.min_y)

    @property
    def max_point(self):
        return Point2D(self.max_x, self.max_y)

    @staticmethod
    def combine(boxes):
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for box in boxes:
            min_x = min(box.min_x, min_x)
            min_y = min(box.min_y, min_y)
            max_x = max(box.max_x, max_x)
            max_y = max(box.max_y, max_y)
        return BoundingBox(min_x, min_y, max_x, max_y)

    def expand(self, radius):
        return BoundingBox(self.min_x - radius, self.min_y - radius, self.max_x + radius, self.max_y + radius)

    @staticmethod
    def from_points(points):
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for pt in points:
            min_x = min(pt[CX], min_x)
            min_y = min(pt[CY], min_y)
            max_x = max(pt[CX], max_x)
            max_y = max(pt[CY], max_y)
        return BoundingBox(min_x, min_y, max_x, max_y)

    def has_point(self, pt):
        return in_range(pt[CX], self.min_x, self.max_x) and in_range(pt[CY], self.min_y, self.max_y)

    def add_points(self, points):
        bad = [point for point in points if not self.has_point(point)]
        if not bad:
            return self
        else:
            return BoundingBox.from_points([self.min_point, self.max_point] + bad)

    def intersects(self, other):
        if self.min_x > other.max_x or other.min_x > self.max_x:
            return False
        if self.min_y > other.max_y or other.min_y > self.max_y:
            return False
        return True

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y
