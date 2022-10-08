import sys

def export(item):
    mod = sys.modules[item.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(item.__name__)
    else:
        mod.__all__ = [item.__name__]
    return item
export(export)
