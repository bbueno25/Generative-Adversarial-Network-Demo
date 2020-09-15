"""
DOCSTRING
"""
import numpy
import os
import rng
import sklearn.externals
import theano_utils
from time

class Constant:
    """
    DOCSTRING
    """
    def __init__(self, c=0.0):
        self.c = c

    def __call__(self, shape):
        return theano_utils.sharedX(numpy.ones(shape) * self.c)

class Frob:
    """
    DOCSTRING
    """
    def __call__(self, shape, name=None):
        r = rng.np_rng.normal(loc=0, scale=0.01, size=shape)
        r = r/numpy.sqrt(numpy.sum(r**2))*numpy.sqrt(shape[1])
        return theano_utils.sharedX(r, name=name)

class Identity:
    """
    DOCSTRING
    """
    def __init__(self, scale=0.25):
        self.scale = scale

    def __call__(self, shape):
        return theano_utils.sharedX(numpy.identity(shape[0]) * self.scale)

class Normal:
    """
    DOCSTRING
    """
    def __init__(self, loc=0., scale=0.05):
        self.scale = scale
        self.loc = loc

    def __call__(self, shape, name=None):
        return theano_utils.sharedX(rng.np_rng.normal(
            loc=self.loc, scale=self.scale, size=shape), name=name)

class Orthogonal:
    """
    benanne lasagne ortho init (faster than qr approach)
    """
    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, shape, name=None):
        flat_shape = (shape[0], numpy.prod(shape[1:]))
        a = rng.np_rng.normal(0.0, 1.0, flat_shape)
        u, _, v = numpy.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return theano_utils.sharedX(self.scale * q[:shape[0], :shape[1]], name=name)

class ReluInit:
    """
    DOCSTRING
    """
    def __call__(self, shape):
        if len(shape) == 2:
            scale = numpy.sqrt(2.0 / shape[0])
        elif len(shape) == 4:
            scale = numpy.sqrt(2.0 / numpy.prod(shape[1:]))
        else:
            raise NotImplementedError
        return theano_utils.sharedX(rng.np_rng.normal(size=shape, scale=scale))

class Uniform:
    """
    DOCSTRING
    """
    def __init__(self, scale=0.05):
        self.scale = 0.05

    def __call__(self, shape):
        return theano_utils.sharedX(
            rng.np_rng.uniform(low=-self.scale, high=self.scale, size=shape))

class W2VEmbedding:
    """
    DOCSTRING
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, vocab, name=None):
        t = time.time()
        w2v_vocab = sklearn.externals.joblib.load(
            os.path.join(self.data_dir, '3m_w2v_gn_vocab.jl'))
        w2v_embed = sklearn.externals.joblib.load(
            os.path.join(self.data_dir, '3m_w2v_gn.jl'))
        mapping = {}
        for i, w in enumerate(w2v_vocab):
            w = w.lower()
            if w in mapping:
                mapping[w].append(i)
            else:
                mapping[w] = [i]
        widxs, w2vidxs = list(), list()
        for i, w in enumerate(vocab):
            w = w.replace('`', "'")
            if w in mapping:
                w2vi = min(mapping[w])
                w2vidxs.append(w2vi)
                widxs.append(i)
        w = numpy.zeros((len(vocab), w2v_embed.shape[1]))
        w[widxs, :] = w2v_embed[w2vidxs, :] / 2.0
        return theano_utils.sharedX(w, name=name)
