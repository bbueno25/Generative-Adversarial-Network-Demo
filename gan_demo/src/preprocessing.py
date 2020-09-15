"""
DOCSTRING
"""
import numpy
import string
import utils

punctuation = set(string.punctuation)
punctuation.add('\n')
punctuation.add('\t')
punctuation.add('')

punc = set(string.punctuation)
punc.add('\t')
punc.add('\n')
punc.remove("'")
contractions = ["'s", "'t", "'re", "'ve", "'ll", "'m", "'d"]

def flatten(l):
    """
    DOCSTRING
    """
    return [item for sublist in l for item in sublist]

def lbf(l, b):
    """
    DOCSTRING
    """
    return [el for el, condition in zip(l, b) if condition]

def list_index(l, idxs):
    """
    DOCSTRING
    """
    return [l[idx] for idx in idxs]

def merge_tokens(tokens):
    """
    DOCSTRING
    """
    merged = [tokens[0]]
    for t in tokens[1:]:
        m = merged[-1]
        if t in punctuation and m[-1] == t:
            merged[-1] += t
            m += t
        elif m.count(m[0]) == len(m) and len(m) > 1 and m[0] in punctuation:
            merged[-1] = m[:4]
            merged.append(t)
        else:
            merged.append(t)
    return merged

def one_hot(X, n=None, negative_class=0.0):
    """
    DOCSTRING
    """
    X = numpy.asarray(X).flatten()
    if n is None:
        n = numpy.max(X) + 1
    Xoh = numpy.ones((len(X), n)) * negative_class
    Xoh[numpy.arange(len(X)), X] = 1.
    return Xoh

def token_encoder(texts, character=False, max_features=9997, min_df=10):
    """
    DOCSTRING
    """
    df = {}
    for text in texts:
        if character:
            text = list(text)
        else:
            text = tokenize(text)
        tokens = set(text)
        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1
    k, v = df.keys(), numpy.asarray(df.values())
    valid = v >= min_df
    k = lbf(k, valid)
    v = v[valid]
    sort_mask = numpy.argsort(v)[::-1]
    k = list_index(k, sort_mask)[:max_features]
    v = v[sort_mask][:max_features]
    xtoi = dict(zip(k, range(3, len(k)+3)))
    return xtoi

def tokenize(text):
    """
    DOCSTRING
    """
    tokenized = list()
    for c in punc:
        text = text.replace(c, ' '+c+' ')
    for c in contractions:
        text = text.replace(c, ' |'+c)
    text = text.replace(" |'", " `")
    text = text.replace("'", " ' ")
    tokens = text.split(' ')
    tokens = [token for token in tokens if token]
    return tokens

def tokenize2(text):
    """
    DOCSTRING
    """
    tokenized = list()
    text = text.replace('\n', ' \n ')
    text = text.replace('\t', ' \t ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.replace(':', ' : ')
    text = text.replace(';', ' ; ')
    text = text.replace('!', ' ! ')
    text = text.replace('?', ' ? ')
    text = text.replace('>', ' > ')
    text = text.replace('<', ' < ')
    text = text.replace('"', ' " ')
    text = text.replace("(", ' ( ')
    text = text.replace(")", ' ) ')
    text = text.replace("[", ' [ ')
    text = text.replace("]", ' ] ')
    text = text.replace("'s", " 's")
    text = text.replace("'t", " 't")
    text = text.replace("'re"," 're")
    text = text.replace("'ve"," 've")
    text = text.replace("'ll"," 'll")
    text = text.replace("'m"," 'm")
    text = text.replace("'d"," 'd")
    text = text.replace(" '", " `")
    text = text.replace("'", " ' ")
    tokens = text.split(' ')
    tokens = [token for token in tokens if token]
    return tokens

def standardize_X(shape, X):
    """
    DOCSTRING
    """
    if not utils.numpy_array(X):
        X = numpy.asarray(X)
    if len(shape) == 4 and len(X.shape) == 2:
        return X.reshape(-1, shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2)
    else:
        return X

def standardize_Y(shape, Y):
    """
    DOCSTRING
    """
    if not utils.numpy_array(Y):
        Y = numpy.asarray(Y)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if len(Y.shape) == 2 and len(shape) == 2:
        if shape[-1] != Y.shape[-1]:
            return one_hot(Y, n=shape[-1])
        else:
            return Y
    else:
        return Y

class LenFilter:
    """
    DOCSTRING
    """
    def __init__(self, max_len=1000, min_max_len=100, percentile=99):
        self.max_len = max_len
        self.percentile = percentile
        self.min_max_len = min_max_len

    def filter(self, *data):
        """
        DOCSTRING
        """
        lens = [len(seq) for seq in data[0]]
        if self.percentile > 0:
            max_len = numpy.percentile(lens, self.percentile)
            max_len = numpy.clip(max_len, self.min_max_len, self.max_len)
        else:
            max_len = self.max_len
        valid_idxs = [i for i, l in enumerate(lens) if l <= max_len]
        if len(data) == 1:
            return list_index(data[0], valid_idxs)
        else:
            return tuple([list_index(d, valid_idxs) for d in data])

class Tokenizer:
    """
    For converting lists of text into tokens used by Passage models.
    max_features sets the maximum number of tokens (all others are mapped to UNK)
    min_df sets the minimum number of documents a token must appear in to not get mapped to UNK
    lowercase controls whether the text is lowercased or not
    character sets whether the tokenizer works on a character or word level

    Usage:
    >>> from passage.preprocessing import Tokenizer
    >>> example_text = ['This. is.', 'Example TEXT', 'is text']
    >>> tokenizer = Tokenizer(min_df=1, lowercase=True, character=False)
    >>> tokenized = tokenizer.fit_transform(example_text)
    >>> tokenized
    [[7, 5, 3, 5], [6, 4], [3, 4]]
    >>> tokenizer.inverse_transform(tokenized)
    ['this . is .', 'example text', 'is text']
    """
    def __init__(self, max_features=9997, min_df=10, lowercase=True, character=False):
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.character = character

    def fit(self, texts):
        """
        DOCSTRING
        """
        if self.lowercase:
            texts = [text.lower() for text in texts]
        self.encoder = token_encoder(
            texts, character=self.character,
            max_features=self.max_features-3, min_df=self.min_df)
        self.encoder['PAD'] = 0
        self.encoder['END'] = 1
        self.encoder['UNK'] = 2
        self.decoder = dict(zip(self.encoder.values(), self.encoder.keys()))
        self.n_features = len(self.encoder)
        return self

    def fit_transform(self, texts):
        """
        DOCSTRING
        """
        self.fit(texts)
        tokens = self.transform(texts)
        return tokens

    def inverse_transform(self, codes):
        """
        DOCSTRING
        """
        if self.character:
            joiner = ''
        else:
            joiner = ' '
        return [joiner.join([self.decoder[token] for token in code]) for code in codes]

    def transform(self, texts):
        """
        DOCSTRING
        """
        if self.lowercase:
            texts = [text.lower() for text in texts]
        if self.character:
            texts = [list(text) for text in texts]
        else:
            texts = [tokenize(text) for text in texts]
        tokens = [[self.encoder.get(token, 2) for token in text] for text in texts]
        return tokens
