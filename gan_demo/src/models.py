"""
DOCSTRING
"""
import activations
import async_iterators
import costs
import iterators
import numpy 
import ops
import sys
import theano
import theano_utils
import time
import utils

def collect_cost(model):
    """
    DOCSTRING
    """
    cost = 0
    for op in model[1:]:
        if hasattr(op, 'cost'):
            cost += op.cost()
    return cost

def collect_infer_updates(model):
    """
    DOCSTRING
    """
    updates = list()
    for op in model[1:]:
        if hasattr(op, 'infer_update'):
            updates.extend(op.infer_update)
    return updates

def collect_reset_updates(model):
    """
    DOCSTRING
    """
    updates = list()
    for op in model[1:]:
        if hasattr(op, 'reset_update'):
            updates.extend(op.reset_update())
    return updates

def collect_updates(model, cost):
    """
    DOCSTRING
    """
    updates = list()
    for op in model[1:]:
        if hasattr(op, 'update'):
            updates.extend(op.update(cost))
    return updates

def init(model):
    """
    DOCSTRING
    """
    print(model[0].out_shape)
    for i in range(1, len(model)):
        model[i].connect(model[i-1])
        if hasattr(model[i], 'init'):
            model[i].init()
    return model

class AdversarialNetwork:
    """
    DOCSTRING
    """
    def __init__(self, model, e, a=0.5, verbose=2, iterator='linear'):
        self.verbose = verbose
        self.model = init(model)
        try:
            self.iterator = utils.instantiate(iterators, iterator)
        except:
            self.iterator = utils.instantiate(async_iterators, iterator)
        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X
        self.Y = theano.tensor.TensorType(
            theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = theano.tensor.nnet.categorical_crossentropy(y_tr, self.Y).mean()
        X_adv = self.X + e*theano.tensor.sgn(theano.tensor.grad(cost, self.X))
        self.model[0].X = X_adv
        y_tr_adv = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        cost_adv = a * cost + (1.0 - a) * theano.tensor.nnet.categorical_crossentropy(
            y_tr_adv, self.Y).mean()
        te_cost = theano.tensor.nnet.categorical_crossentropy(y_te, self.Y).mean()
        X_te_adv = self.X + e*theano.tensor.sgn(theano.tensor.grad(te_cost, self.X))
        self.updates = collect_updates(self.model, cost_adv)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.Y], cost_adv, updates=self.updates)
        self._predict = theano.function([self.X], y_te)
        self._fast_sign = theano.function([self.X, self.Y], X_te_adv)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trY, n_iter=1):
        """
        DOCSTRING
        """
        out_shape = self.model[-1].out_shape
        n = 0.
        t = time.time()
        cost = list()
        for e in range(n_iter):
            epoch_costs = list()
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time.time() - t)
                    n_left = len(trY)*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write(
                        "\rIter %d Seen %d samples Avg cost %0.4f" \
                            "Examples per second %d Time left %d seconds" % (
                                e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            cost.extend(epoch_costs)
            n_per_sec = n / (time.time() - t)
            n_left = len(trY)*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f" \
                "Examples per second %d Time elapsed %d seconds" % (
                    e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time.time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print(status)
        return cost

    def infer(self, X):
        """
        DOCSTRING
        """
        self._reset()
        if isinstance(self.iterator, iterators.Linear):
            return self.infer_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.infer_idxs(X)
        else:
            raise NotImplementedError

    def infer_idxs(self, X):
        """
        DOCSTRING
        """
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._infer(xmb)

    def infer_iterator(self, X):
        """
        DOCSTRING
        """
        for xmb in self.iterator.iterX(X):
            self._infer(xmb)

    def predict(self, X):
        """
        DOCSTRING
        """
        if isinstance(self.iterator, iterators.Linear):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_idxs(self, X):
        preds, idxs = list(), list()
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return numpy.vstack(preds)[numpy.argsort(idxs)]

    def predict_iterator(self, X):
        """
        DOCSTRING
        """
        preds = list()
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        return numpy.vstack(preds)

class EmbeddingNetwork:
    """
    DOCSTRING
    """
    def __init__(self, model, iterator, alpha=0.2, verbose=2):
        self.model = init(model)
        self.iterator = iterator
        self.alpha = alpha
        self.verbose = verbose
        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X
        anchor = y_tr[::3]
        pos = y_tr[1::3]
        neg = y_tr[2::3]
        dpos = theano_utils.pair_euclidean(anchor, pos)
        dneg = theano_utils.pair_euclidean(anchor, neg)
        d = dneg - dpos
        cost = theano.tensor.maximum((1. - self.alpha) - d, 0.).mean()
        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X], cost, updates=self.updates)
        self._transform = theano.function([self.X], y_te)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, n_iter=1):
        """
        DOCSTRING
        """
        n = 0.
        t = time.time()
        cost = list()
        for e in range(n_iter):
            epoch_costs = list()
            for xmb in self.iterator.train(trX):
                c = self._train(xmb)
                epoch_costs.append(c)
                n += xmb.shape[self.model[0].out_shape.index('x')]/3
                if self.verbose >= 2:
                    n_per_sec = n / (time.time() - t)
                    n_left = self.iterator.batches*self.iterator.size*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write(
                        "\rIter %d Seen %d samples Avg cost %0.4f" \
                        "Examples per second %d Time left %d seconds" % (
                            e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            cost.extend(epoch_costs)
            n_per_sec = n / (time.time() - t)
            n_left = self.iterator.batches*self.iterator.size*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f" \
                "Examples per second %d Time elapsed %d seconds" % (
                    e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time.time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print(status)
        return cost

    def infer(self, X):
        """
        DOCSTRING
        """
        self._reset()
        for xmb in self.iterator.predict(X):
            self._infer(xmb)

    def transform(self, X):
        """
        DOCSTRING
        """
        Xt = list()
        for xmb in self.iterator.predict(X):
            xt = self._transform(xmb)
            Xt.append(xt)
        return numpy.vstack(Xt)

class Network:
    """
    DOCSTRING
    """
    def __init__(self, model, cost=None, verbose=2, iterator='linear'):
        if cost is not None:
            self.cost = utils.instantiate(costs, cost)
        else:
            if isinstance(model[-1], ops.Activation):
                if isinstance(model[-1].activation, activations.Sigmoid):
                    self.cost = utils.instantiate(costs, 'bce')
                elif isinstance(model[-1].activation, activations.Softmax):
                    self.cost = utils.instantiate(costs, 'cce')
                else:
                    self.cost = utils.instantiate(costs, 'mse')
            else:
                self.cost = utils.instantiate(costs, 'mse')
        self.verbose = verbose
        self.model = init(model)
        try:
            self.iterator = utils.instantiate(iterators, iterator)
        except:
            self.iterator = utils.instantiate(async_iterators, iterator)
        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X
        self.Y = theano.tensor.TensorType(
            theano.config.floatX, (False,)*(len(model[-1].out_shape)))()
        cost = self.cost(self.Y, y_tr)
        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X, self.Y], cost, updates=self.updates)
        self._predict = theano.function([self.X], y_te)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, trY, n_iter=1):
        """
        DOCSTRING
        """
        out_shape = self.model[-1].out_shape
        n = 0.
        t = time.time()
        cost = list()
        for e in range(n_iter):
            epoch_costs = list()
            for xmb, ymb in self.iterator.iterXY(trX, trY):
                c = self._train(xmb, ymb)
                epoch_costs.append(c)
                n += len(ymb)
                if self.verbose >= 2:
                    n_per_sec = n / (time.time() - t)
                    n_left = len(trY)*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write(
                        "\rIter %d Seen %d samples Avg cost %0.4f" \
                            "Examples per second %d Time left %d seconds" % (
                                e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            cost.extend(epoch_costs)
            n_per_sec = n / (time.time() - t)
            n_left = len(trY)*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f" \
                "Examples per second %d Time elapsed %d seconds" % (
                    e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time.time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print(status)
        return cost

    def infer_iterator(self, X):
        """
        DOCSTRING
        """
        for xmb in self.iterator.iterX(X):
            self._infer(xmb)

    def infer_idxs(self, X):
        """
        DOCSTRING
        """
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._infer(xmb)

    def infer(self, X):
        """
        DOCSTRING
        """
        self._reset()
        if isinstance(self.iterator, iterators.Linear):
            return self.infer_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.infer_idxs(X)
        else:
            raise NotImplementedError

    def predict(self, X):
        """
        DOCSTRING
        """
        if isinstance(self.iterator, iterators.Linear):
            return self.predict_iterator(X)
        elif isinstance(self.iterator, iterators.SortedPadded):
            return self.predict_idxs(X)
        else:
            raise NotImplementedError

    def predict_iterator(self, X):
        """
        DOCSTRING
        """
        preds = list()
        for xmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
        return numpy.vstack(preds)

    def predict_idxs(self, X):
        """
        DOCSTRING
        """
        preds, idxs = list(), list()
        for xmb, idxmb in self.iterator.iterX(X):
            pred = self._predict(xmb)
            preds.append(pred)
            idxs.extend(idxmb)
        return numpy.vstack(preds)[numpy.argsort(idxs)]

class SimNetwork:
    """
    DOCSTRING
    """
    def __init__(self, model, iterator, verbose=2):
        self.model = init(model)
        self.iterator = iterator
        self.verbose = verbose
        y_tr = self.model[-1].op({'dropout':True, 'bn_active':True, 'infer':False})
        y_te = self.model[-1].op({'dropout':False, 'bn_active':False, 'infer':False})
        y_inf = self.model[-1].op({'dropout':False, 'bn_active':True, 'infer':True})
        self.X = self.model[0].X
        cos_sim = cosine(y_tr[::4], y_tr[1::4])
        cos_diff = cosine(y_tr[2::4], y_tr[3::4])
        cost = theano.tensor.mean(theano.tensor.maximum(0, 1 - cos_sim + cos_diff))
        self.updates = collect_updates(self.model, cost)
        self.infer_updates = collect_infer_updates(self.model)
        self.reset_updates = collect_reset_updates(self.model)
        self._train = theano.function([self.X], cost, updates=self.updates)
        self._transform = theano.function([self.X], y_te)
        self._infer = theano.function([self.X], y_inf, updates=self.infer_updates)
        self._reset = theano.function([], updates=self.reset_updates)

    def fit(self, trX, n_iter=1):
        """
        DOCSTRING
        """
        n = 0.
        t = time.time()
        cost = list()
        for e in range(n_iter):
            epoch_costs = list()
            for xmb in self.iterator.train(trX):
                c = self._train(xmb)
                epoch_costs.append(c)
                n += xmb.shape[self.model[0].out_shape.index('x')]/4
                if self.verbose >= 2:
                    n_per_sec = n / (time.time() - t)
                    n_left = self.iterator.batches*self.iterator.size*n_iter - n
                    time_left = n_left/n_per_sec
                    sys.stdout.write(
                        "\rIter %d Seen %d samples Avg cost %0.4f" \
                            "Examples per second %d Time left %d seconds" % (
                                e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time_left))
                    sys.stdout.flush()
            cost.extend(epoch_costs)
            n_per_sec = n / (time.time() - t)
            n_left = self.iterator.batches*self.iterator.size*n_iter - n
            time_left = n_left/n_per_sec
            status = "Iter %d Seen %d samples Avg cost %0.4f" \
                "Examples per second %d Time elapsed %d seconds" % (
                    e, n, numpy.mean(epoch_costs[-250:]), n_per_sec, time.time() - t)
            if self.verbose >= 2:
                sys.stdout.write("\r"+status) 
                sys.stdout.flush()
                sys.stdout.write("\n")
            elif self.verbose == 1:
                print(status)
        return cost

    def infer(self, X):
        """
        DOCSTRING
        """
        self._reset()
        for xmb in self.iterator.predict(X):
            self._infer(xmb)

    def transform(self, X):
        """
        DOCSTRING
        """
        Xt = list()
        for xmb in self.iterator.predict(X):
            xt = self._transform(xmb)
            Xt.append(xt)
        return numpy.vstack(Xt)
