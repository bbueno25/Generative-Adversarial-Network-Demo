"""
DOCSTRING
"""
import theano
import theano_utils

def clip_norm(g, c, n):
    """
    DOCSTRING
    """
    if c > 0:
        g = theano.tensor.switch(theano.tensor.ge(n, c), g * c / n, g)
    return g

def clip_norms(gs, c):
    """
    DOCSTRING
    """
    norm = theano.tensor.sqrt(sum([theano.tensor.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]

class Adadelta(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.5, rho=0.95, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        grads = theano.tensor.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_delta = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc,acc_new))
            update = g * theano.tensor.sqrt(
                acc_delta + self.epsilon) / theano.tensor.sqrt(acc_new + self.epsilon)
            updated_p = p - self.lr * update
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
            acc_delta_new = self.rho * acc_delta + (1 - self.rho) * update ** 2
            updates.append((acc_delta,acc_delta_new))
        return updates

class Adagrad(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        grads = theano.tensor.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.0)
            acc_t = acc + g ** 2
            updates.append((acc, acc_t))
            p_t = p - (self.lr / theano.tensor.sqrt(acc_t + self.epsilon)) * g
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((p, p_t))
        return updates  

class Adam(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost, consider_constant=None):
        updates = list()
        grads = theano.tensor.grad(cost, params, consider_constant=consider_constant)
        grads = clip_norms(grads, self.clipnorm)  
        t = theano.shared(theano_utils.floatX(1.0))
        b1_t = self.b1 * self.l**(t-1)
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.0)
            v = theano.shared(p.get_value() * 0.0)
            m_t = b1_t * m + (1 - b1_t) * g
            v_t = self.b2 * v + (1 - self.b2)*g**2
            m_c = m_t / (1 - self.b1**t)
            v_c = v_t / (1 - self.b2**t)
            p_t = p - (self.lr * m_c) / (theano.tensor.sqrt(v_c) + self.e)
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((t, t + 1.0))
        return updates

class Momentum(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        grads = theano.tensor.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m, v))
            updated_p = p + v
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

class NoUpdate(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        for p in params:
            updates.append((p, p))
        return updates

class NAG(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        grads = theano.tensor.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updated_p = p + self.momentum * v - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((m,v))
            updates.append((p, updated_p))
        return updates

class Regularizer:
    """
    DOCSTRING
    """
    def __init__(self, l1=0.0, l2=0.0, maxnorm=0.0, l2norm=False, frobnorm=False):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        """
        DOCSTRING
        """
        if maxnorm > 0:
            norms = theano.tensor.sqrt(theano.tensor.sum(theano.tensor.sqr(p), axis=0))
            desired = theano.tensor.clip(norms, 0, maxnorm)
            p = p * (desired/ (1e-7 + norms))
        return p

    def l2_norm(self, p, axis=0):
        """
        DOCSTRING
        """
        return p/theano_utils.l2norm(p, axis=axis)

    def frob_norm(self, p, nrows):
        """
        DOCSTRING
        """
        return (p / theano.tensor.sqrt(theano.tensor.sum(
            theano.tensor.sqr(p)))) * theano.tensor.sqrt(nrows)

    def gradient_regularize(self, p, g):
        """
        DOCSTRING
        """
        g += p * self.l2
        g += theano.tensor.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        """
        DOCSTRING
        """
        p = self.max_norm(p, self.maxnorm)
        if self.l2norm:
            p = self.l2_norm(p, self.l2norm)
        if self.frobnorm > 0:
            p = self.frob_norm(p, self.frobnorm)
        return p

class RMSprop(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        grads = theano.tensor.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_new))
            updated_p = p - self.lr * (g / theano.tensor.sqrt(acc_new + self.epsilon))
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

class SGD(Update):
    """
    DOCSTRING
    """
    def __init__(self, lr=0.01, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def __call__(self, params, cost):
        updates = list()
        grads = theano.tensor.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            updated_p = p - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

class Update:
    """
    DOCSTRING
    """
    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        raise NotImplementedError
