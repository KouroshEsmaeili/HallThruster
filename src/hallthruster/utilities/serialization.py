
import numbers
from collections import OrderedDict

# ––– Marker Classes –––

class SType:
    """Base marker type for serialization traits."""
    pass

class Null(SType):
    pass

class Boolean(SType):
    pass

class NumberType(SType):
    pass

class StringType(SType):
    pass

class ArrayType(SType):
    pass

class TupleType(SType):
    pass

class EnumType(SType):
    pass

# Composite marker types
class Composite(SType):
    pass

class TaggedUnion(Composite):
    pass

class Struct(Composite):
    pass

# ––– Helper functions for trait‐based dispatch –––

def SType_of(T):
    # Note: in Python, bool is a subclass of int, so check bool first.
    if T is bool:
        return Boolean()
    if T is type(None):
        return Null()
    if issubclass(T, numbers.Number):
        return NumberType()
    if issubclass(T, str):
        return StringType()
    if T is tuple:
        return TupleType()
    # For arrays we accept list and tuple (but tuple is already caught above)
    if issubclass(T, list):
        return ArrayType()
    # For Symbol in Julia we use str (or simply treat as String)
    # Otherwise, assume a composite structure.
    return Struct()

def exclude(T):
    return []

def options(T):
    return {}

def typetag(T):
    return "type"

def iterate_fields(T):
    excl = exclude(T)
    ann = getattr(T, '__annotations__', {})
    for name, ftype in ann.items():
        if name not in excl:
            yield name, ftype


def serialize(x):
    return _serialize(SType_of(type(x)), x)

def deserialize(T, x):
    return _deserialize(SType_of(T), T, x)

# ––– Dispatch functions –––

def _serialize(marker, x):
    if isinstance(marker, Null):
        return None
    elif isinstance(marker, Boolean):
        return bool(x)
    elif isinstance(marker, NumberType):
        return x
    elif isinstance(marker, StringType):
        return str(x)
    elif isinstance(marker, ArrayType):
        # Assume x is an iterable; serialize each element.
        return [serialize(elem) for elem in x]
    elif isinstance(marker, TupleType):
        return tuple(serialize(elem) for elem in x)
    elif isinstance(marker, Struct):
        # Assume x is an instance of a composite class.
        result = OrderedDict()
        for field, ftype in iterate_fields(type(x)):
            result[field] = serialize(getattr(x, field))
        return result
    elif isinstance(marker, TaggedUnion):
        opts = options(type(x))
        # Look for a key k so that type(x) is a subclass of opts[k].
        for k, subtype in opts.items():
            if issubclass(type(x), subtype):
                # Serialize all fields.
                result = OrderedDict()
                result[typetag(type(x))] = str(k)
                for field, _ in iterate_fields(type(x)):
                    result[field] = serialize(getattr(x, field))
                return result
        raise ValueError(f"Invalid type {type(x)}. Valid options are {list(opts.keys())}")
    elif isinstance(marker, EnumType):
        opts = options(type(x))
        for k, v in opts.items():
            if v == x:
                return k
        raise ValueError(f"Invalid value {x} for type {type(x)}. Valid options are {list(opts.keys())}")
    else:
        # Fallback: return x as is.
        return x

def _deserialize(marker, T, x):
    if isinstance(marker, Null):
        return None
    elif isinstance(marker, Boolean):
        return T(x)
    elif isinstance(marker, NumberType):
        return T(x)
    elif isinstance(marker, StringType):
        return T(x)
    elif isinstance(marker, ArrayType):
        # Assume T is list-like; deserialize each element.
        return [deserialize(type(elem), elem) for elem in x]
    elif isinstance(marker, TupleType):
        return tuple(deserialize(type(elem), elem) for elem in x)
    elif isinstance(marker, Struct):
        # Assume x is an OrderedDict (or dict) with field values.
        kwargs = {}
        for field, ftype in iterate_fields(T):
            kwargs[field] = deserialize(ftype, x[field])
        return T(**kwargs)
    elif isinstance(marker, TaggedUnion):
        tag = typetag(T)
        tag_val = x.get(tag)
        opts = options(T)
        subtype = opts.get(tag_val)
        if subtype is None:
            raise ValueError(f"Unknown tag {tag_val} for type {T}")
        kwargs = {}
        # For each field in the dict except the type tag, deserialize according to subtype.
        for field in x:
            if field != tag:
                # Assume subtype has __annotations__
                ftype = getattr(subtype, '__annotations__', {}).get(field)
                if ftype is None:
                    kwargs[field] = x[field]
                else:
                    kwargs[field] = deserialize(ftype, x[field])
        return subtype(**kwargs)
    elif isinstance(marker, EnumType):
        opts = options(T)
        # Here x is expected to be the key; return the corresponding enum value.
        return opts.get(x)
    else:
        return x  # Fallback

# ––– Example Usage –––
if __name__ == "__main__":
    # Define a simple composite type.
    class Point:
        __annotations__ = {'x': float, 'y': float}
        def __init__(self, *, x, y):
            self.x = x
            self.y = y
        def __repr__(self):
            return f"Point(x={self.x}, y={self.y})"

    pt = Point(x=1.0, y=2.0)
    ser = serialize(pt)
    print("Serialized Point:", ser)
    pt2 = deserialize(Point, ser)
    print("Deserialized Point:", pt2)

    # Example for arrays.
    arr = [1, 2, 3]
    ser_arr = serialize(arr)
    print("Serialized array:", ser_arr)
    arr2 = deserialize(list, ser_arr)
    print("Deserialized array:", arr2)
