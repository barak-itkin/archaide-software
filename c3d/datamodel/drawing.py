from .utils import JsonSerializable
from .outline import Outline


class Drawing(JsonSerializable):
    """A drawing is a dictionary of named outlines."""
    def __init__(self, outlines={}, properties={}):
        self.outlines = dict(
            (key, val.clone()) for key, val in dict(outlines).items()
        )
        self.properties = dict(properties)

    def deserialize(self, data):
        self.__init__(
            outlines=dict(
                (key, Outline().deserialize(val)) for key, val in data.get('outlines', {}).items()
            ),
            properties=dict(data.get('properties', {}))
        )
        return self

    def serialize(self):
        return {
            'outlines': dict(
                (key, val.serialize()) for key, val in self.outlines.items()
            ),
            'properties': dict(self.properties)
        }

    def keys(self):
        return self.outlines.keys()

    def items(self):
        return self.outlines.items()

    def __contains__(self, item):
        return item in self.outlines

    def __getitem__(self, item):
        return self.outlines[item]
