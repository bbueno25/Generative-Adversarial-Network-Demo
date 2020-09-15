"""
DOCSTRING
"""
import numpy
import rng
import utils

y(Xt).transpose(0, 2, 1)[:, :, :, numpy.newaxis]

def CenterCrop(X, pw, ph):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        w, h = x.shape[:2]
        i = int(round((w-pw)/2.))
        j = int(round((h-ph)/2.))
        Xt.append(x[i:i+pw, j:j+pw])
    return Xt

def ColorShift(X, p=1/3., scale=20):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        x = x.astype(numpy.int16)
        x[:, :, 0] += (rng.py_rng.random() < p) * np_rng.py_rng.randint(-scale, scale)
        x[:, :, 1] += (rng.py_rng.random() < p) * np_rng.py_rng.randint(-scale, scale)
        x[:, :, 2] += (rng.py_rng.random() < p) * np_rng.py_rng.randint(-scale, scale)
        x = numpy.clip(x, 0, 255).astype(numpy.uint8)
        Xt.append(x)
    return Xt

def FlatToImg(X, w, h, c):
    """
    DOCSTRING
    """
    if not utils.numpy_array(X):
        X = numpy.asarray(X)
    return X.reshape(-1, w, h, c)

def FlipHorizontal(X):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        if rng.py_rng.random() > 0.5:
            x = numpy.fliplr(x)
        Xt.append(x)
    return Xt

def Fliplr(X):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        if rng.py_rng.random() > 0.5:
            x = numpy.fliplr(x)
        Xt.append(x)
    return Xt

def FlipVertical(X):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        if rng.py_rng.random() > 0.5:
            x = numpy.flipud(x)
        Xt.append(x)
    return Xt

def ImgToConv(X):
    """
    DOCSTRING
    """
    if not utils.numpy_array(X):
        X = numpy.asarray(X)
    return X.transpose(0, 3, 1, 2)

def LenClip(X, n):
    """
    DOCSTRING
    """
    Xc = list()
    for x in X:
        words = x.split(' ')
        lens = [len(word) + 1 for word in words]
        lens[0] -= 1
        lens[-1] -= 1
        lens = numpy.cumsum(lens).tolist()
        words = [w for w, l in zip(words, lens) if l < n]
        xc = ' '.join(words)
        Xc.append(xc)
    return Xc

def MorphTokenize(X, encoder, max_encoder_len):
    """
    DOCSTRING
    """
    Xt = list()
    for n, text in enumerate(X):
        tokens, i = list(), 0
        while i < len(text):
            for l in range(max_encoder_len)[::-1]:
                if text[i:i+l+1] in encoder:
                    tokens.append(text[i:i+l+1])
                    i += l+1
                    break
            if l == 0:
                tokens.append(text[i])
                i += 1
        Xt.append(tokens)
    return Xt

def OneHot(X, n=None, negative_class=0.0):
    """
    DOCSTRING
    """
    X = numpy.asarray(X).flatten()
    if n is None:
        n = numpy.max(X) + 1
    Xoh = numpy.ones((len(X), n)) * negative_class
    Xoh[numpy.arange(len(X)), X] = 1.
    return Xoh

def Patch(X, pw, ph):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X: 
        w, h = x.shape[:2]
        i = rng.py_rng.randint(0, w-pw)
        j = rng.py_rng.randint(0, h-ph)
        Xt.append(x[i:i+pw, j:j+pw])
    return Xt

def Reflect(X):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        if rng.py_rng.random() > 0.5:
            x = numpy.flipud(x)
        if rng.py_rng.random() > 0.5:
            x = numpy.fliplr(x)
        Xt.append(x)
    return Xt

def Rot90(X):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        x = numpy.rot90(x, rng.py_rng.randint(0, 3))
        Xt.append(x)
    return Xt

def SeqDelete(X, p_delete):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        Xt.append([w for w in x if rng.py_rng.random() > p_delete])
    return Xt

def SeqPadded(seqs):
    """
    DOCSTRING
    """
    lens = map(len, seqs)
    max_len = max(lens)
    seqs_padded = list()
    for seq, seq_len in zip(seqs, lens):
        n_pad = max_len - seq_len 
        seq = [0] * n_pad + seq
        seqs_padded.append(seq)
    return numpy.asarray(seqs_padded).transpose(1, 0)

def SeqPatch(X, p_size):
    """
    DOCSTRING
    """
    Xt = list()
    for x in X:
        l = len(x)
        n = int(p_size*l)
        i = rng.py_rng.randint(0, l-n)
        Xt.append(x[i:i+n])
    return Xt

def Standardize(X):
    """
    DOCSTRING
    """
    if not utils.numpy_array(X):
        X = numpy.asarray(X)
    return X / 127.5 - 1.0

def StringToCharacterCNNRep(X, max_len, encoder):
    """
    DOCSTRING
    """
    nc = len(encoder) + 1
    Xt = list()
    for x in X:
        x = [encoder.get(c, 2) for c in x]
        x = OneHot(x, n=nc)
        l = len(x)
        if l != max_len:
            x = numpy.concatenate([x, numpy.zeros((max_len-l, nc))])
        Xt.append(x)
    return numpy.asarra

def StringToCharacterCNNIDXRep(X, max_len, encoder):
    """
    DOCSTRING
    """
    nc = len(encoder) + 1
    Xt = list()
    for x in X:
        x = [encoder.get(c, 2) for c in x]
        l = len(x)
        if l != max_len:
            x = numpy.concatenate([x, numpy.zeros((max_len-l))])
        Xt.append(x)
    return numpy.asarray(Xt).transpose(1, 0)

def StringToCharacterCNNRNNRep(X, encoder):
    """
    DOCSTRING
    """
    nc = len(encoder) + 1
    Xt = list()
    max_len = max([len(x) for x in X])
    for x in X:
        x = [encoder.get(c, 0) for c in x]
        x = one_hot(x, n=nc)
        x = [encoder.get(c, 2) for c in x]
        x = OneHot(x, n=nc)
        l = len(x)
        if l != max_len:
            x = numpy.concatenate([numpy.zeros((max_len-l, nc)), x])
        Xt.append(x)
    return numpy.asarray(Xt).reshape(len(Xt), 1, max_len, nc)

def ZeroOneScale(X):
    """
    DOCSTRING
    """
    if not utils.numpy_array(X):
        X = numpy.asarray(X)
    return X / 255.0
