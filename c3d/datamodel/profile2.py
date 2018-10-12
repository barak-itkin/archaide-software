from .outline2 import Outline2
from .utils import JsonSerializable


class Profile2(JsonSerializable):
    """A profile2 is a list of 2D Outline2.

    Additionally, profiles hold the area of the largest triangle eliminated in
    the outline simplification process."""
    def __init__(self, outlines=(), simplification_area=0):
        self.outlines = [o.clone() for o in outlines]
        self.simplification_area = simplification_area

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

    def serialize(self):
        return {
            'outlines': [o.serialize() for o in self.outlines],
            'simplification_area': self.simplification_area
        }
