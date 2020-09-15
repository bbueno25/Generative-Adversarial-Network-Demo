"""
DOCSTRING
"""
import numpy
import theano

def cosine(x, y):
    """
    DOCSTRING
    """
    d = theano.tensor.dot(x, y.theano.tensor)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d

def diag_gaussian(X, m, c):
    """
    theano version of function from sklearn/mixture/gmm/gmm.py
    """
    n_samples, n_dim = X.shape
    lpr = -0.5 * (
        n_dim * numpy.log(2 * numpy.pi)
        + theano.tensor.sum(theano.tensor.log(c), 1)
        + theano.tensor.sum((m ** 2) / c, 1)
        - 2 * theano.tensor.dot(X, (m / c).theano.tensor)
        + theano.tensor.dot(X ** 2, (1.0 / c).theano.tensor))
    return lpr

def downcast_float(X):
    """
    DOCSTRING
    """
    return numpy.asarray(X, dtype=numpy.float32)

def euclidean(x, y, e1=1e-3, e2=1e-3):
    """
    DOCSTRING
    """
    xx = theano.tensor.sqr(theano.tensor.sqrt((x * x).sum(axis=1) + e1))
    yy = theano.tensor.sqr(theano.tensor.sqrt((y * y).sum(axis=1) + e1))
    dist = theano.tensor.dot(x, y.theano.tensor)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = theano.tensor.sqrt(dist + e2)
    return dist

def floatX(X):
    """
    DOCSTRING
    """
    return numpy.asarray(X, dtype=theano.config.floatX)

def intX(X):
    """
    DOCSTRING
    """
    return numpy.asarray(X, dtype=numpy.int32)

def l1norm(x, axis=1):
    """
    DOCSTRING
    """
    return theano.tensor.sum(theano.tensor.abs_(x), axis=axis)

def l2norm(x, axis=1, e=1e-8):
    """
    DOCSTRING
    """
    return theano.tensor.sqrt(
        theano.tensor.sum(theano.tensor.sqr(x), axis=axis, keepdims=True) + e)

def pair_cosine(a, b, e=1e-8):
    """
    DOCSTRING
    """
    return theano.tensor.sum(a*b, axis=1)/(l2norm(a, e=e)*l2norm(b, e=e))

def pair_euclidean(a, b, axis=1, e=1e-8):
    """
    DOCSTRING
    """
    return theano.tensor.sqrt(
        theano.tensor.sum(theano.tensor.sqr(a - b), axis=axis) + e)

def shared0s(shape, dtype=theano.config.floatX, name=None):
    """
    DOCSTRING
    """
    return sharedX(numpy.zeros(shape), dtype=dtype, name=name)

def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    """
    DOCSTRING
    """
    return sharedX(numpy.ones(shape)*n, dtype=dtype, name=name)

def sharedX(X, dtype=theano.config.floatX, name=None):
    """
    DOCSTRING
    """
    return theano.shared(numpy.asarray(X, dtype=dtype), name=name)
