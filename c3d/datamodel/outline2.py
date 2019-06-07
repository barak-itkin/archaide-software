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

    def optimize_unused(self):
        # For every three consecutive points with the two segments between as
        # NEITHER, we can eliminate the middle point (and the segment from it)
        # as no one is ever going to refer to it. This will reduce both the file
        # size and all the compute time later on these profiles.
        n_pts, n_segments = len(self.points), len(self.inside_map)

        if n_segments == 0:
            return self

        skip_points = set(
            i for i in range(1, n_segments)
            if all(s == self.NEITHER for s in self.inside_map[i - 1:i + 1])
        )

        if self.is_closed:
            assert n_segments == n_pts
            # For a closed path, also check the cycle point
            if self.inside_map[-1] == self.inside_map[0] == self.NEITHER:
                skip_points.add(0)
            skip_segments = skip_points
        else:
            assert n_segments == n_pts - 1
            # For an open path, remove NEITHER segments from both sides
            if self.inside_map[0] == self.NEITHER:
                skip_points.add(0)
            skip_segments = set(skip_points)
            if self.inside_map[-1] == self.NEITHER:
                skip_points.add(n_segments)
                skip_segments.add(n_segments - 1)

        if len(skip_points) == 0:
            return self

        return Outline2(
            points=(p for i, p in enumerate(self.points)
                    if i not in skip_points),
            inside_map=(s for i, s in enumerate(self.inside_map)
                        if i not in skip_segments),
        )

    def transform(self, pts_transform):
        return Outline2(
            points=(pts_transform(p) for p in self.points),
            inside_map=self.inside_map
        )

    def scale(self, factor):
        return self.transform(lambda p: (p[0] * factor, p[1] * factor))

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
