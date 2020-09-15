"""
DOCSTRING
"""
import theano

class ClippedRectify:
    """
    DOCSTRING
    """
    def __init__(self, clip=10.):
        self.clip = clip

    def __call__(self, x):
        return theano.tensor.clip((x + abs(x)) / 2.0, 0., self.clip)

class ConvRMSPool:
    """
    DOCSTRING
    """
    def __call__(self, x):
        x = x**2
        return theano.tensor.sqrt(x[:,::2,:,:] + x[:,1::2,:,:] + 1e-8)

class ConvSoftmax:
    """
    DOCSTRING
    """
    def __call__(self, x):
        e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
class ELU:
    """
    DOCSTRING
    """
    def __call__(self, x):
        return theano.tensor.switch(theano.tensor.ge(x, 0), x, theano.tensor.exp(x)-1)
    
class HardSigmoid:
    """
    DOCSTRING
    """
    def __call__(self, X):
        return theano.tensor.clip(X + 0.5, 0.0, 1.0)
    
class LeakyRectify:
    """
    DOCSTRING
    """
    def __init__(self, leak=0.2):
        self.leak = leak

    def __call__(self, x):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * x + f2 * abs(x)

class Linear:
    """
    DOCSTRING
    """
    def __call__(self, x):
        return x

class MaskedConvSoftmax:
    """
    DOCSTRING
    """
    def __call__(self, x, m):
        x = x*m.dimshuffle(0, 'x', 1, 'x')
        e_x = theano.tensor.exp(x - x.max(axis=2, keepdims=True))
        e_x = e_x*m.dimshuffle(0, 'x', 1, 'x')
        return e_x / e_x.sum(axis=2, keepdims=True)

class Maxout:
    """
    DOCSTRING
    """
    def __init__(self, n_pool=2):
        self.n_pool = n_pool

    def __call__(self, x):
        if x.ndim == 2:
            x = theano.tensor.max([x[:, n::self.n_pool] for n in range(self.n_pool)], axis=0)
        elif x.ndim == 4:
            x = theano.tensor.max([x[:, n::self.n_pool, :, :] for n in range(self.n_pool)], axis=0)
        elif x.ndim == 3:
            print('assuming standard rnn 3tensor')
            x = theano.tensor.max([x[:, :, n::self.n_pool] for n in range(self.n_pool)], axis=0)
        return x
    
class Prelu:
    """
    DOCSTRING
    """
    def __call__(self, x, leak):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        if leak.ndim == 1:
            return theano.tensor.flatten(f1, 1)[0] * x + theano.tensor.flatten(f2, 1)[0] * abs(x)
        else:
            return f1 * x + f2 * abs(x)

class Rectify:
    """
    DOCSTRING
    """
    def __call__(self, x):
        return (x + abs(x)) / 2.0

class Sigmoid:
    """
    DOCSTRING
    """
    def __call__(self, x):
        return theano.tensor.nnet.sigmoid(x)

class Softmax:
    """
    DOCSTRING
    """
    def __call__(self, x):
        e_x = theano.tensor.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

class SteeperSigmoid:
    """
    DOCSTRING
    """
    def __init__(self, scale=3.75):
        self.scale = scale

    def __call__(self, x):
        return 1.0 / (1.0 + theano.tensor.exp(-self.scale * x))

class Tanh:
    """
    DOCSTRING
    """
    def __call__(self, x):
        return theano.tensor.tanh(x)
