from .utils import JsonSerializable


class Fracture(JsonSerializable):
    """A fracture is a cycle of 3D vertices around a face, and a face normal."""
    def __init__(self, vertices=[], normal=(0, 0, 0)):
        self.vertices = [
            (v[0], v[1], v[2]) for v in vertices
            ]
        self.normal = list(normal)

    def deserialize(self, data):
        self.__init__(vertices=data['vertices'], normal=data['normal'])
        return self

    def serialize(self):
        return {
            'vertices': self.vertices,
            'normal': self.normal,
        }
