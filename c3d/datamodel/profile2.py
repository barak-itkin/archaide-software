from typing import List

from .outline2 import Outline2
from .utils import JsonSerializable


class Profile2(JsonSerializable):
    """A profile2 is a list of 2D Outline2.

    Additionally, profiles hold the area of the largest triangle eliminated in
    the outline simplification process."""
    def __init__(self, outlines=(), simplification_area=0):
        self.outlines = [o.clone() for o in outlines]  # type: List[Outline2]
        self.simplification_area = simplification_area  # type: float

    def __len__(self):
        return len(self.outlines)

    def __iter__(self):
        return iter(self.outlines)

    def __getitem__(self, item):
        return self.outlines[item]

    def deserialize(self, data):
        self.__init__(
            outlines=[Outline2().deserialize(o) for o in data['outlines']],
            simplification_area=data['simplification_area']
        )
        return self

    def optimize_unused(self):
        return Profile2(
            outlines=[o.optimize_unused() for o in self.outlines],
            simplification_area=self.simplification_area
        )

    def scale(self, factor):
        return Profile2(
            outlines=[o.scale(factor) for o in self.outlines],
            simplification_area=self.simplification_area * factor * factor,
        )

    def serialize(self):
        return {
            'outlines': [o.serialize() for o in self.outlines],
            'simplification_area': self.simplification_area
        }
