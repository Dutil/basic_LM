import numpy as np
import ipdb
from theano import tensor as T, config
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
    #ipdb.set_trace()
    return -1* sum([np.dot(p, [np.math.log(i, 2) for i in q]) for p, q in zip(ps, qs)])


def save_everything(saving_path, rnn, metadata):

    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    print "saving to... {}".format(saving_path)

    rnn_file_name = os.path.join(saving_path, "rnn.pkl")
    metadata_name = os.path.join(saving_path, "metadata")

    rnn.save(rnn_file_name)
    pickle.dump(metadata, open(metadata_name, 'w'))

def load_everything(loading_path):


    rnn_file_name = os.path.join(loading_path, "rnn.pkl")
    metadata_name = os.path.join(loading_path, "metadata")

    # ATTENTION, LSTM ne marche pas, je sais.
    #ipdb.set_trace()
    rnn = None

    #if is_LSTM:
    #    rnn = RNN.LSTM()
    #else:
    #    rnn = RNN.RNN()
    rnn = RNN.MLP()

    rnn.load(rnn_file_name)

    metadata = pickle.load(open(metadata_name))

    return rnn, metadata


def hotify_minibatch(minibatch, v_size, pad_before=1):
    """
    Makes sure all the sentences in the minibatch are the same length. Also add an empty word at the beginning.
    Plus make the sentences 1-hot.
    :param minibatch: a list of sentences
    :return: a padded list of sentences.
    """
    max_len = max([len(x) for x in minibatch])
    sentences = []

    # ipdb.set_trace()
    for sentence in minibatch:
        sentence = oneHots(sentence, v_size)  # one hot representation
        sentence = np.pad(sentence, ((pad_before, max_len - len(sentence)), (0, 0)),
                          'constant', constant_values=(0))  # padding to the max length
        sentences.append(sentence)

    sentences = np.array(sentences).astype(config.floatX)
    sentences = sentences.transpose((1, 0, 2))
    return sentences