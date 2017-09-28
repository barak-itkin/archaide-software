from .outline import Outline
from .utils import JsonSerializable


class Profile(JsonSerializable):
    """A profile is a 2D outline, with a switch index (between in and out).

    Additionally, profiles hold the area of the largest triangle eliminated in
    the outline simplification process."""
    def __init__(self, outline=Outline(), switch_index=0, simplification_area=0):
        self.outline = outline.clone()
        self.switch_index = switch_index
        self.simplification_area = simplification_area

    def deserialize(self, data):
        self.__init__(
            outline=Outline().deserialize(data['outline']),
            switch_index=data['switch_index'],
            simplification_area=data['simplification_area']
        )
        return self

    def serialize(self):
        return {
            'outline': self.outline.serialize(),
            'switch_index': self.switch_index,
            'simplification_area': self.simplification_area
        }
