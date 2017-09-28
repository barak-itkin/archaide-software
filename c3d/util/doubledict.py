class DoubleDict:
    def __init__(self, initializer=()):
        self.forward = {}
        self.reverse = {}

        if hasattr(initializer, 'items') and callable(initializer.items):
            initializer = initializer.items()

        for key, value in iter(initializer):
            self[key] = value

    def __contains__(self, item):
        return item in self.forward

    def __getitem__(self, item):
        return self.forward[item]

    def __setitem__(self, key, value):
        if value in self.reverse:
            raise KeyError('Value already exists: %s' % value)
        if key in self.forward:
            del self.reverse[self.forward[key]]
        self.forward[key] = value
        self.reverse[value] = key

    def __delitem__(self, key):
        value = self.forward[key]
        del self.forward[key]
        del self.reverse[value]

    def __bool__(self):
        return bool(self.forward)

    def __len__(self):
        return len(self.forward)

    def items(self):
        return self.forward.items()

    def keys(self):
        return self.forward.keys()

    def values(self):
        return self.forward.values()

    def rename(self, old_key, new_key):
        value = self.forward[old_key]
        del self[old_key]
        self[new_key] = value

    # Implement pickle support1
    def __getstate__(self):
        return self.forward

    def __setstate__(self, state):
        self.__init__(state.items())
