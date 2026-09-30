
def keywords(cls):
    annotations = getattr(cls, '__annotations__', {})
    if not annotations:
        raise ValueError("Invalid usage of @keywords: class must have type annotations.")

    # Obtain default values defined at the class level.
    class_defaults = {
        k: getattr(cls, k)
        for k in annotations.keys()
        if hasattr(cls, k)
    }

    def __init__(self, **kwargs):
        for field, field_type in annotations.items():
            # Use the keyword argument if given; otherwise, try the class default.
            if field in kwargs:
                value = kwargs[field]
            elif field in class_defaults:
                value = class_defaults[field]
            else:
                raise TypeError(f"Missing required keyword argument: {field}")
            # Attempt conversion via the annotated type.
            try:
                # If the value is already of the right type, this should be harmless.
                value = field_type(value)
            except Exception:
                # If conversion fails, leave the value as is.
                pass
            setattr(self, field, value)

    # Optionally, preserve any existing __init__ signature as well.
    __init__.__doc__ = f"Auto-generated __init__ for {cls.__name__}.\n" \
                         "Accepts only keyword arguments corresponding to the annotated fields."
    cls.__init__ = __init__
    return cls

# Example usage:
if __name__ == "__main__":
    @keywords
    class Example:
        a: float
        b: int = 42
        c: str = "default"

    # Create an instance by supplying keyword arguments.
    e = Example(a=7, c="hello")
    print("a =", e.a)  # prints 7.0 because a is converted to float
    print("b =", e.b)  # prints 42 (the default)
    print("c =", e.c)  # prints "hello"
