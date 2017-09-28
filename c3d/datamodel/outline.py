from .utils import JsonSerializable


class Outline(JsonSerializable):
    """An outline is a path of 2D points."""
    def __init__(self, points=[]):
        self.points = [
            (p[0], p[1]) for p in points
        ]

    def deserialize(self, data):
        self.__init__(points=data)
        return self

    def serialize(self):
        return list(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)