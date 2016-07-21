import numpy as np
import ipdb
from theano import tensor as T
import os, pickle, RNN

def oneHot( word, nb_class):
        onehot = np.zeros(nb_class, dtype=np.float32)
        onehot[word] = 1
        return onehot

def oneHots(xs, nb_class):
    return np.array([oneHot(x, nb_class) for x in xs]).astype(np.float32)


def t_crossEntropy(p, q):
    return - T.sum(p * T.log(q))

def crossEntropy(ps, qs):
    return - sum([np.dot(p, [np.math.log(i, 2) for i in q]) for p, q in zip(ps, qs)])


def save_everything(saving_path, rnn, metadata):

    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    print "saving to... {}".format(saving_path)

    rnn_file_name = os.path.join(saving_path, "rnn.pkl")
    metadata_name = os.path.join(saving_path, "metadata")

    rnn.save(rnn_file_name)
    pickle.dump(metadata, open(metadata_name, 'w'))

def load_everything(loading_path, is_LSTM):


    rnn_file_name = os.path.join(loading_path, "rnn.pkl")
    metadata_name = os.path.join(loading_path, "metadata")

    # ATTENTION, LSTM ne marche pas, je sais.
    #ipdb.set_trace()
    rnn = None

    if is_LSTM:
        rnn = RNN.LSTM()
    else:
        rnn = RNN.RNN()

    rnn.load(rnn_file_name)

    metadata = pickle.load(open(metadata_name))

    return rnn, metadata