"""
DOCSTRING
"""
import numpy.random
import random
import theano.sandbox.rng_mrg

np_rng = numpy.random.RandomState(seed)
py_rng = random.Random(seed)
seed = 42
t_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

def set_seed(n):
    """
    DOCSTRING
    """
    seed = n
    py_rng = random.Random(seed)
    np_rng = numpy.random.RandomState(seed)
    t_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)
