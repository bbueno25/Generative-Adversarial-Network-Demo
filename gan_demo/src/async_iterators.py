"""
DOCSTRING
"""
import multiprocessing
import theano_utils
import utils

def noop(x):
    """
    DOCSTRING
    """
    return x

class AsyncLinear:
    """
    size is the number of examples per minibatch
    shuffle controls whether or not the order of examples is shuffled before iterating over
    x_dtype is for casting input data
    y_dtype is for casting target data
    """
    def __init__(
        self,
        size=128,
        shuffle=True,
        train_load=noop,
        train_transform=noop,
        test_load=noop,
        test_transform=noop):
        self.size = size
        self.shuffle = shuffle
        self.train_load = train_load
        self.train_transform = train_transform
        self.test_load = test_load
        self.test_transform = test_transform

    def iterX(self, X):
        """
        DOCSTRING
        """
        self.loader = Loader(X, self.test_load, self.test_transform, self.size)
        self.proc = multiprocessing.Process(target=self.loader.load)
        self.proc.start()
        for xmb in utils.iter_data(X, size=self.size):
            xmb = self.loader.get()
            yield xmb

    def iterXY(self, X, Y):
        """
        DOCSTRING
        """
        if self.shuffle:
            X, Y = utils.shuffle(X, Y)
        self.loader = Loader(X, self.train_load, self.train_transform, self.size)
        self.proc = multiprocessing.Process(target=self.loader.load)
        self.proc.start()
        for ymb in utils.iter_data(Y, size=self.size):
            xmb = self.loader.get()
            yield xmb, theano_utils.floatX(ymb)

class Loader:
    """
    DOCSTRING
    """
    def __init__(self, X, load_fn, transform_fn, size, max_batches=5):
        self.X = X
        self.load_fn = load_fn
        self.transform_fn = transform_fn
        self.size = size
        self.batches = multiprocessing.Queue(max_batches)

    def get(self):
        """
        DOCSTRING
        """
        return self.transform_fn(self.batches.get())
    
    def load(self):
        """
        DOCSTRING
        """
        pool = multiprocessing.Pool(8)
        for xmb in utils.iter_data(self.X, size=self.size):
            xmb = pool.map(self.load_fn, xmb)
            self.batches.put(xmb)
        self.batches.put(StopIteration)
