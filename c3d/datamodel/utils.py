"""Utilities to assist with de/serialization of data objects."""


from functools import wraps
import json


def with_file_handle(mode, instance):
    """Decorate a function to take either a path or a file handle.

    The decorate function always receive a file handle, regardless of which
    parameter was passed in practice. If indeed a file handle was open, the
    decorator will also close it at the end of the function execution."""
    def decorator(func):
        if instance:
            @wraps(func)
            def result(self, path_or_handle, *args, **kwargs):
                if isinstance(path_or_handle, str):
                    with open(path_or_handle, mode) as f:
                        return func(self, f, *args, **kwargs)
                else:
                    return func(self, path_or_handle, *args, **kwargs)
        else:
            @wraps(func)
            def result(path_or_handle, *args, **kwargs):
                if isinstance(path_or_handle, str):
                    with open(path_or_handle, mode) as f:
                        return func(f, *args, **kwargs)
                else:
                    return func(path_or_handle, *args, **kwargs)
        return result
    return decorator


def with_read_handle(instance):
    return with_file_handle('r', instance)


def with_write_handle(instance):
    return with_file_handle('w', instance)


class JsonSerializable:
    """Base-class for data objects supporting JSON de/serialization."""
    def serialize(self):
        """Return the JSON serializable primitive representing this object."""
        raise NotImplementedError()

    def deserialize(self, data):
        """Load the object from JSON deserialized data AND RETURN SELF!"""
        raise NotImplementedError()

    def dumps(self, *args, **kwargs):
        return json.dumps(self.serialize(), *args, **kwargs)

    def loads(self, value, *args, **kwargs):
        self.deserialize(json.loads(value, *args, **kwargs))

    @with_write_handle(True)
    def dump(self, fp, *args, **kwargs):
        json.dump(self.serialize(), fp, *args, **kwargs)

    @with_read_handle(True)
    def load(self, fp, *args, **kwargs):
        self.deserialize(json.load(fp, *args, **kwargs))

    def clone(self):
        copy = type(self)()
        copy.deserialize(self.serialize())
        return copy


@with_read_handle(False)
def load_json(f, type, *args, **kwargs):
    result = type()
    result.load(f, *args, **kwargs)
    return result


def loads_json(value, type, *args, **kwargs):
    result = type()
    result.loads(value, *args, **kwargs)
    return result
