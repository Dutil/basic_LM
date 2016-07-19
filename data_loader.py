from collections import Counter
import numpy as np
import ipdb

class data_iterator:

    def __init__(self, data, e_size, m_size, vocab, wordMapping):

        self.data = data.split("\n")
        self.m_size = m_size
        self.vocab = vocab
        self.wordMapping = wordMapping
        self.e_size = e_size

        self.noEx = 0 # The no of the exemple we are presenty at
        self.nbMinibatch = 0 # The number of minibatch we have done right now

    def __iter__(self):

        minibatch = []
        for i in self.data[self.noEx:]:

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
        self.e_size = 0

    def switchRep(self, ids):
        return [self.wordMapping[x] for x in ids]

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
        mostCommon["--OOV--"] = sum(lessCommon.values())
        wordMapping["--OOV--"] = nbWords
        wordMapping[nbWords] = "--OOV--"

        self.vocab = mostCommon
        self.wordMapping = wordMapping
        self.nbWords = nbWords + 1

    def replaceOOV(self, data):

        # replace the less frequent words to OOV
        data = data.split()
        for noWord, word in enumerate(data):
            if word not in self.vocab:
                data[noWord] = "--OOV--"
        data = " ".join(data)
        return data





