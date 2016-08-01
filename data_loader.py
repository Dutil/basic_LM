from collections import Counter
import numpy as np
import ipdb, utils


class predict_next_iterator:

    def __init__(self, iterator):
        self.iterator = iterator
        self.v_size = len(iterator.vocab)


    def __iter__(self):

        for minibatch in self.iterator:
            minibatch = utils.hotify_minibatch(minibatch, self.v_size)
            yield minibatch[:-1], minibatch[1:]

class predict_noisy_self:
    def __init__(self, iterator, p=0.0):
        self.iterator = iterator
        self.v_size = len(iterator.vocab)
        self.p = p


    def __iter__(self):

        for minibatch in self.iterator:

            noisy_minibatch = [filter(lambda x: np.random.rand() >= self.p, m) for m in minibatch]

            hot_minibatch = utils.hotify_minibatch(minibatch, self.v_size, pad_before=0)
            hot_noisy_minibatch = utils.hotify_minibatch(noisy_minibatch, self.v_size, pad_before=0)



            yield hot_noisy_minibatch, hot_minibatch

class data_iterator:

    def __init__(self, data, e_size, m_size, vocab, wordMapping):

        self.data = data.split("\n")
        np.random.seed(1993)
        np.random.shuffle(self.data)

        self.m_size = m_size
        self.vocab = vocab
        self.wordMapping = wordMapping
        self.e_size = e_size

        self.noEx = 0 # The no of the exemple we are presenty at
        self.nbMinibatch = 0 # The number of minibatch we have done right now

    def __iter__(self):

        minibatch = []
        for i in self.data[self.noEx:]:

            if i: # for empty sentences
                minibatch.append(self.switchRep(i.split()))

            if len(minibatch) >= self.m_size:

                self.noEx += len(minibatch)
                self.nbMinibatch += 1

                yield minibatch
                minibatch = []

            #If we consider that we have done one epoch
            if self.nbMinibatch == self.e_size:
                self.nbMinibatch = 0
                raise StopIteration

        #Left over exemples
        if minibatch:
            yield minibatch

        self.noEx = 0
        self.nbMinibatch = 0

    def switchRep(self, ids):
        sentence = [self.wordMapping[x] if not type(x) == np.ndarray
                else self.wordMapping[np.where(x)[0][0]] if sum(x) > 0 else '' for x in ids]

        return filter(None, sentence)

    def __getitem__(self, key):
        return self.switchRep(self.data[key].split())

class data_crawler:

    def __init__(self, folder="testing_data", maxCount = 10):

        self.folder = folder
        self.maxCount = maxCount

        training_set = open("{}/train.txt".format(folder)).read()
        valid_set = open("{}/valid.txt".format(folder)).read()
        test_set = open("{}/test.txt".format(folder)).read()
        self.all_data = [training_set, valid_set, test_set]

        self._initVocab(" ".join(self.all_data))


    def _initVocab(self, data):

        vocab = Counter()
        wordMapping = {}
        nbWords = 0

        #Get the vocab
        #data = data.replace("\n", "--EOS--")
        vocab.update(data.split())

        #The words we are keeping
        mostCommon = Counter(dict(vocab.most_common(self.maxCount - 1)))
        lessCommon = vocab - mostCommon


        #get the words ids
        for i, word in enumerate(mostCommon):
            wordMapping[word] = i
            wordMapping[i] = word
            nbWords = i + 1

        #The OOV
        #mostCommon["--OOV--"] = sum(lessCommon.values())
        #wordMapping["--OOV--"] = nbWords
        #wordMapping[nbWords] = "--OOV--"

        self.vocab = mostCommon
        self.wordMapping = wordMapping
        self.nbWords = nbWords #+ 1

    def replaceOOV(self, data):

        # replace the less frequent words to OOV
        data = data.split()
        for noWord, word in enumerate(data):
            if word not in self.vocab:
                data[noWord] = "--OOV--"
        data = " ".join(data)
        return data





