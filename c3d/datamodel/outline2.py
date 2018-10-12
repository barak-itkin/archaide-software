from .utils import JsonSerializable


class Outline2(JsonSerializable):
    """An Outline2 is a 2D/3D outline, where the inside_map marks whether the segment
       starting at the matching point is inside, outside, or neither."""
    INSIDE = 1
    OUTSIDE = 2
    NEITHER = 0

    def __init__(self, points=(), inside_map=()):
        self.points = [
            tuple(p) for p in points
        ]
        self.inside_map = list(inside_map)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, item):
        return self.points[item]

    @property
    def is_closed(self):
        return len(self.inside_map) == len(self.points)

    @property
    def is_open(self):
        return len(self.inside_map) < len(self.points)

    def reversed(self):
        if self.is_open:
            return Outline2(reversed(self.points), reversed(self.inside_map))
        else:
            return Outline2(
                #       2
                #       *        pts:     0  1  2
                #      / \       map:     a  b  c
                #     c   b
                #    /     \     rev-pts: 2  1  0
                #   *---a---*    map:     b  a  c
                #  0         1
                #
                list(reversed(self.points)),
                list(reversed(self.inside_map[:-1])) + [self.inside_map[-1]]
            )

    def transform(self, pts_transform):
        return Outline2(
            points=(pts_transform(p) for p in self.points),
            inside_map=self.inside_map
        )

    def deserialize(self, data):
        self.__init__(
            points=data['points'],
            inside_map=data['inside_map'],
        )
        return self

    def serialize(self):
        return {
            'points': self.points,
            'inside_map': self.inside_map,
        }
