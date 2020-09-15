"""
DOCSTRING
"""
import theano

cce = CCE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError


def BinaryCrossEntropy(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano.tensor.nnet.binary_crossentropy(y_pred, y_true).mean()

def CategoricalCrossEntropy(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano.tensor.nnet.categorical_crossentropy(y_pred, y_true).mean()

def Hinge(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano.tensor.maximum(1. - y_true * y_pred, 0.).mean()

def MeanAbsoluteError(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano.tensor.abs_(y_pred - y_true).mean()

def MeanSquaredError(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano.tensor.sqr(y_pred - y_true).mean()

def PairEuclidean(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano_utils.pair_euclidean(y_pred, y_true).mean()

def SquaredHinge(y_true, y_pred):
    """
    DOCSTRING
    """
    return theano.tensor.sqr(theano.tensor.maximum(1. - y_true * y_pred, 0.)).mean()
