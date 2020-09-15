"""
DOCSTRING
"""
import inspect
import numpy
import rng
import sklearn
import types

def case_insensitive_import(module, name):
    """
    DOCSTRING
    """
    mapping = dict((k.lower(), k) for k in dir(module))
    return getattr(module, mapping[name.lower()])

def classes_of(module):
    """
    DOCSTRING
    """
    return tuple(x[1] for x in inspect.getmembers(module, inspect.isclass))

def instantiate(module, obj):
    """
    DOCSTRING
    """
    if isinstance(obj, basestring):
        obj = case_insensitive_import(module, obj)
        if isinstance(obj, types.FunctionType):
            return obj
        else:
            return obj()
    elif isinstance(obj, classes_of(module)):
    	return obj
    elif inspect.isfunction(obj):
    	return obj
    else:
        raise TypeError

def iter_data(*data, **kwargs):
    """
    DOCSTRING
    """
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data]) 

def iter_indices(*data, **kwargs):
    """
    DOCSTRING
    """
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1
    for b in range(batches):
        yield b

def list_shuffle(*data):
    """
    DOCSTRING
    """
    idxs = rng.np_rng.permutation(numpy.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]
    
def numpy_array(X):
    """
    DOCSTRING
    """
    return type(X).__module__ == numpy.__name__

def shuffle(*arrays, **options):
    """
    DOCSTRING
    """
    if isinstance(arrays[0][0], basestring):
        return list_shuffle(*arrays)
    else:
        return sklearn.utils.shuffle(*arrays, random_state=rng.np_rng)
