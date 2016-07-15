from collections import Counter
import numpy as np


class data_crawler:

    def __init__(self, path="data.txt", in_memory = True, maxCount = 10):

        self.path = path
        self.data = open(self.path)
        self.maxCount = maxCount

        if in_memory:
            self.data = self.data.read().split("\n")

        self.metadata = {}
        self._initVocab()


    def __iter__(self):

        for i in self.data:
            yield self.switchRep(i.split(" "))

    def __getitem__(self, key):
        return self.switchRep(self.data[key].split(" "))

    def _initVocab(self):

        vocab = Counter()
        wordMapping = {}
        nbWords = 0

        #Get the vocab
        for i in self.data:
            vocab.update(i.split(" "))

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

        #replace the less frequent words to OOV
        for noSentence, sentence in enumerate(self.data):
            sentence = sentence.split(" ")
            for noWord, word in enumerate(sentence):
                if word in lessCommon:
                    sentence[noWord] = "--OOV--"
            self.data[noSentence] = " ".join(sentence)

        self.metadata["vocab"] = mostCommon
        self.metadata['wordMapping'] = wordMapping
        self.metadata['nbWords'] = nbWords + 1

    def switchRep(self, ids):
        return [self.metadata["wordMapping"][x] for x in ids]








