"""
DOCSTRING
"""
import foxhound
import matplotlib.pyplot as pyplot
import numpy
import scipy.misc
import theano

batch_size = 128
bce = theano.tensor.nnet.binary_crossentropy
init_fn = foxhound.inits.Normal(scale=0.02)
leakyrectify = foxhound.activations.LeakyRectify()
nh = 2048
rectify = foxhound.activations.Rectify()
sigmoid = foxhound.activations.Sigmoid()
tanh = foxhound.activations.Tanh()

def d(X, w, g, b, w2, g2, b2, wo):
    """
    Build Discriminator Network
    """
    h = rectify(scale_and_shift(theano.tensor.dot(X, w), g, b))
    h2 = tanh(scale_and_shift(theano.tensor.dot(h, w2), g2, b2))
    y = sigmoid(theano.tensor.dot(h2, wo))
    return y

def g(X, w, g, b, w2, g2, b2, wo):
    """
    Build Generator Network
    """
    h = leakyrectify(scale_and_shift(theano.tensor.dot(X, w), g, b))
    h2 = leakyrectify(scale_and_shift(theano.tensor.dot(h, w2), g2, b2))
    y = theano.tensor.dot(h2, wo)
    return y

def gaussian_likelihood(X, u=0.0, s=1.0):
    """
    Visualize Gaussian Curves
    """
    return (1.0 / (s * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((X - u)**2) / (2 * s**2)))

def scale_and_shift(X, g, b, e=1e-8):
    """
    DOCSTRING
    """
    return X * g + b

def vis(i):
    """
    plotting function - plot both curves (for G and D)
    """
    s, u = 1.0, 0.0
    zs = numpy.linspace(-1, 1, 500).astype('float32')
    xs = numpy.linspace(-5, 5, 500).astype('float32')
    ps = gaussian_likelihood(xs, 1.0)
    gs = _gen(zs.reshape(-1, 1)).flatten()
    preal = _score(xs.reshape(-1, 1)).flatten()
    kde = scipy.stats.gaussian_kde(gs)
    pyplot.clf()
    pyplot.plot(xs, ps, '--', lw=2)
    pyplot.plot(xs, kde(xs), lw=2)
    pyplot.plot(xs, preal, lw=2)
    pyplot.xlim([-5., 5.])
    pyplot.ylim([0., 1.])
    pyplot.ylabel('Prob')
    pyplot.xlabel('x')
    pyplot.legend(['P(data)', 'G(z)', 'D(x)'])
    pyplot.title('GAN learning guassian')
    fig.canvas.draw()
    pyplot.show(block=False)
    pyplot.show()

if __name__ == '__main__':
    gw = init_fn((1, nh))
    gg = foxhound.inits.Constant(1.0)(nh)
    gg = foxhound.inits.Normal(1.0, 0.02)(nh)
    gb = foxhound.inits.Normal(0.0, 0.02)(nh)
    gw2 = init_fn((nh, nh))
    gg2 = foxhound.inits.Normal(1.0, 0.02)(nh)
    gb2 = foxhound.inits.Normal(0.0, 0.02)(nh)
    gy = init_fn((nh, 1))
    ggy = foxhound.inits.Constant(1.0)(1)
    gby = foxhound.inits.Normal(0.0, 0.02)(1)
    dw = init_fn((1, nh))
    dg = foxhound.inits.Normal(1.0, 0.02)(nh)
    db = foxhound.inits.Normal(0.0, 0.02)(nh)
    dw2 = init_fn((nh, nh))
    dg2 = foxhound.inits.Normal(1.0, 0.02)(nh)
    db2 = foxhound.inits.Normal(0.0, 0.02)(nh)
    dy = init_fn((nh, 1))
    dgy = foxhound.inits.Normal(1.0, 0.02)(1)
    dby = foxhound.inits.Normal(0.0, 0.02)(1)
    g_params = [gw, gg, gb, gw2, gg2, gb2, gy]
    d_params = [dw, dg, db, dw2, dg2, db2, dy]
    Z = theano.tensor.matrix()
    X = theano.tensor.matrix()
    gen = g(Z, *g_params)
    p_real = d(X, *d_params)
    p_gen = d(gen, *d_params)
    d_cost_real = bce(p_real, theano.tensor.ones(p_real.shape)).mean()
    d_cost_gen = bce(p_gen, theano.tensor.zeros(p_gen.shape)).mean()
    g_cost_d = bce(p_gen, theano.tensor.ones(p_gen.shape)).mean()
    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d
    cost = [g_cost, d_cost, d_cost_real, d_cost_gen]
    lr = 0.001
    lrt = foxhound.theano_utils.sharedX(lr)
    d_updater = foxhound.updates.Adam(lr=lrt)
    g_updater = foxhound.updates.Adam(lr=lrt)
    d_updates = d_updater(d_params, d_cost)
    g_updates = g_updater(g_params, g_cost)
    updates = d_updates + g_updates
    _train_g = theano.function([X, Z], cost, updates=g_updates)
    _train_d = theano.function([X, Z], cost, updates=d_updates)
    _train_both = theano.function([X, Z], cost, updates=updates)
    _gen = theano.function([Z], gen)
    _score = theano.function([X], p_real)
    _cost = theano.function([X, Z], cost)
    fig = pyplot.figure()
    for i in range(100):
        zmb = numpy.random.uniform(-1, 1, size=(batch_size, 1)).astype('float32')
        xmb = numpy.random.normal(1.0, 1, size=(batch_size, 1)).astype('float32')
        if i % 10 == 0:
            _train_g(xmb, zmb)
        else:
            _train_d(xmb, zmb)
        if i % 10 == 0:
            print(i)
            vis(i)
        lrt.set_value(foxhound.theano_utils.floatX(lrt.get_value() * 0.9999))
