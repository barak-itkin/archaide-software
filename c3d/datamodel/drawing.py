from .utils import JsonSerializable
from .outline import Outline


class Drawing(JsonSerializable):
    """A drawing is a dictionary of named outlines."""
    def __init__(self, outlines={}):
        self.outlines = dict(
            (key, val.clone()) for key, val in dict(outlines).items()
        )

    def deserialize(self, data):
        self.__init__(outlines=dict(
            (key, Outline().deserialize(val)) for key, val in data.items()
        ))
        return self

    def serialize(self):
        return dict(
            (key, val.serialize()) for key, val in self.outlines.items()
        )

    def __contains__(self, item):
        return item in self.outlines

    def __getitem__(self, item):
        return self.outlines[item]