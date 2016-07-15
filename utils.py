import numpy as np
import ipdb
from theano import tensor as T

def oneHot( word, nb_class):
        onehot = np.zeros(nb_class, dtype=np.float32)
        onehot[word] = 1
        return onehot

def oneHots(xs, nb_class):
    return np.array([oneHot(x, nb_class) for x in xs]).astype(np.float32)


def crossEntropy(p, q):
    return - T.sum(p * T.log(q))
