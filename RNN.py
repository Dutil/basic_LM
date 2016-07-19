import numpy as np
import theano.tensor as T
import theano
from theano import function, shared, config
import ipdb, utils, time, copy, pickle


class RNN:
    def __init__(self, h_size = 3, e_size = 2, v_size = 10, lr = 0.01):

        self.h_size = h_size
        self.e_size = e_size
        self.v_size = v_size

        self.lr = lr

        np.random.seed(seed=1993)
        self.initParams()

        self.initThenoFunctions()

    def initThenoFunctions(self):
        self.t_fp, self.t_pred = self.getFunc()

    def initParams(self):
        """

        :return: The initial parameters of the RNN
        """

        self.emb = shared(np.asarray(np.random.rand(self.v_size, self.e_size) - 0.5, config.floatX), name="Emb")
        self.Wx = shared(np.asarray(np.random.rand(self.e_size, self.h_size) - 0.5, config.floatX), name="Wx")
        self.Wh = shared(np.asarray(np.random.rand(self.h_size, self.h_size) - 0.5, config.floatX), name="Wh")
        self.Wo = shared(np.asarray(np.random.rand(self.h_size, self.v_size) - 0.5, config.floatX), name="Wo")

        #biais
        self.Whb = shared(np.asarray(np.random.rand(self.h_size) - 0.5, config.floatX), name="Whb")
        self.Wob = shared(np.asarray(np.random.rand(self.v_size) - 0.5, config.floatX), name="Wob")

        self.params = [self.emb, self.Wx, self.Wh, self.Wo, self.Whb, self.Wob]

    def get_outputs_info(self, xs):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((xs.shape[1], self.h_size), config.floatX),# h0
                None, None]# output, loss

    def train(self, nb_epoch, trainingSet, validSet):
        """
        :param nb_epoch: number of epoch to train the model
        :param data: The data over wich we are training
        :return: The losses for each epochs
        """

        trainLosses = []
        validLosses = []
        for i in range(nb_epoch):
            epochTime = time.clock()

            print "doing epoch {}".format(i)
            loss = self.doOneEpoch(trainingSet)
            print "Cost is: {} for the training set".format(loss)
            trainLosses.append(loss)

            loss = self.getLoss(validSet)
            print "Cost is: {} for the valid set".format(loss)
            validLosses.append(loss)
            print "Total epoch time in: {}".format(time.clock() - epochTime)

        return trainLosses, validLosses

    def doOneEpoch(self, data):

        losses = 0.0
        #ipdb.set_trace()
        for minibatch in data:
            sentences = self.hotify_minibatch(minibatch)

            loss = self.forwardPass(sentences)
            losses += loss

        return losses

    def hotify_minibatch(self, minibatch):

        max_len = max([len(x) for x in minibatch])
        sentences = []

        #ipdb.set_trace()
        for sentence in minibatch:
            sentence = utils.oneHots(sentence, self.v_size)# one hot representation
            sentence = np.pad(sentence, ((0, max_len-len(sentence)), (0, 0)),
                              'constant', constant_values=(0))# padding to the max length
            sentences.append(sentence)

        sentences = np.array(sentences).astype(config.floatX)
        sentences = sentences.transpose((1, 0, 2))
        return sentences

    def forwardPass(self, minibatch):

        loss = self.t_fp(minibatch[:-1], minibatch[1:])
        return loss

    def get_hidden_function(self):

        def hidden_function(xt, yt, h_tm1):

            et = T.dot(xt, self.emb)

            # hidden layer
            ht = T.dot(et, self.Wx) + T.dot(h_tm1, self.Wh) + self.Whb
            ht = T.nnet.sigmoid(ht)

            # output
            ot = T.dot(ht, self.Wo) + self.Wob
            ot = T.nnet.softmax(ot)

            # loss
            loss = utils.t_crossEntropy(yt, ot)

            return ht, ot, loss

        return hidden_function

    def getFunc(self):

        xs = T.ftensor3("xs")# (no_seq, no_minibatch, no_word)
        ys = T.ftensor3("ys")

        outputs, updates = theano.scan(fn=self.get_hidden_function(),
                                       outputs_info=self.get_outputs_info(xs),
                                       sequences=[xs, ys])

        lossT = outputs[-1]
        oT = outputs[-2]

        sum_lossT = lossT.sum()
        gParams  = T.grad(sum_lossT, self.params)
        updates = [(p, p - self.lr*gp) for p, gp in zip(self.params, gParams)]

        back_prob = function([xs, ys], sum_lossT, updates=updates) #return the total loss of the minibatch
        prediction = function([xs, ys], [oT, sum_lossT])# return the softmaxes, and the loss for every sentences

        return back_prob, prediction

    def predict(self, sentences):
        """
        For every sentence, for every word xi, predict the word xi+1
        :param sentences: the list of sentences
        :return:
        """

        minibatch = self.hotify_minibatch(sentences)

        pred_softmax, _ = self.t_pred(minibatch, minibatch)#the softmaxes
        pred_softmax = pred_softmax.transpose((1,0,2))
        preds = []

        # Get the word with the maximum probability for each sentences
        for sentence, pred_sentence in zip(sentences, pred_softmax):

            pred = np.zeros(len(sentence)).astype(np.int32)
            for i in range(len(sentence)):
                pred[i] = np.argmax(pred_sentence[i])

            preds.append(pred)

        return preds

    def getPerplexity(self, dataset):
        """
        Compute the perplexity for a particular datasets.

        :param sentences:
        :return:
        """

        perplexity = 0.0
        for minibatch in dataset:
            hot_minibatch = self.hotify_minibatch(minibatch)
            m_xs, m_ys = hot_minibatch[:-1], hot_minibatch[1:]
            m_pred_softmax, _ = self.t_pred(m_xs, m_ys)
            m_pred_softmax = m_pred_softmax.transpose((1, 0, 2))

            average_losses = [utils.crossEntropy(ys, softmax)/len(sentence)
                              for ys, softmax, sentence in zip(m_ys.transpose((1,0,2)), m_pred_softmax, minibatch)]

            #ipdb.set_trace()
            perplexity = perplexity + np.exp2(np.mean(average_losses))

        return perplexity

    def getLoss(self, dataset):
        """
        Get the total loss of a particular dataset

        :param dataset:
        :return:
        """

        totalLoss = 0.0
        for minibatch in dataset:
            hot_minibatch = self.hotify_minibatch(minibatch)
            m_xs, m_ys = hot_minibatch[:-1], hot_minibatch[1:]
            _, loss = self.t_pred(m_xs, m_ys)
            totalLoss += loss

        return totalLoss

    def save(self, path):

        # I know, I know. I have some diplicate, but I don't really care :)
        params = copy.copy(self.__dict__)

        #I don't really care about saving the functions.
        del params["t_fp"]
        del params["t_pred"]
        pickle.dump(params, open(path, 'w'))

    def load(self, path):
        params = pickle.load(open(path))
        self.__dict__.update(params)
        self.initThenoFunctions()





class LSTM(RNN):

    def __init__(self, **params):
        RNN.__init__(self, **params)

    def initParams(self):

        #Embedings
        self.emb = shared(np.asarray(np.random.rand(self.v_size, self.e_size) - 0.5, config.floatX), name="Emb")

        #Inputs gate weights
        self.Wix = shared(np.asarray(np.random.rand(self.e_size, self.h_size) - 0.5, config.floatX), name="Wix")
        self.Wih = shared(np.asarray(np.random.rand(self.h_size, self.h_size) - 0.5, config.floatX), name="Wih")
        self.Wic = shared(np.diag(np.random.rand(self.h_size) - 0.5).astype(config.floatX), name = "Wic")
        self.Wib = shared(np.asarray(np.random.rand(self.h_size) - 0.5, config.floatX), name="Wib")

        #forget gates weights
        self.Wfx = shared(np.asarray(np.random.rand(self.e_size, self.h_size) - 0.5, config.floatX), name="Wfx")
        self.Wfh = shared(np.asarray(np.random.rand(self.h_size, self.h_size) - 0.5, config.floatX), name="Wfh")
        self.Wfc = shared(np.diag(np.random.rand(self.h_size) - 0.5).astype(config.floatX), name = "Wfc")
        self.Wfb = shared(np.asarray(np.random.rand(self.h_size) - 0.5, config.floatX), name="Wfb")

        #output gate weights
        self.Wox = shared(np.asarray(np.random.rand(self.e_size, self.h_size) - 0.5, config.floatX), name="Wox")
        self.Woh = shared(np.asarray(np.random.rand(self.h_size, self.h_size) - 0.5, config.floatX), name="Woh")
        self.Woc = shared(np.diag(np.random.rand(self.h_size) - 0.5).astype(config.floatX), name = "Woc")
        self.Wob = shared(np.asarray(np.random.rand(self.h_size) - 0.5, config.floatX), name="Wob")

        #cell weights
        self.Wcx = shared(np.asarray(np.random.rand(self.e_size, self.h_size) - 0.5, config.floatX), name="Wcx")
        self.Wch = shared(np.diag(np.random.rand(self.h_size) - 0.5).astype(config.floatX), name = "Wch")
        self.Wcb = shared(np.asarray(np.random.rand(self.h_size) - 0.5, config.floatX), name="Wcb")

        #output weights
        self.Wo = shared(np.asarray(np.random.rand(self.h_size, self.v_size) - 0.5, config.floatX), name="Wo")
        self.Woutb = shared(np.asarray(np.random.rand(self.v_size) - 0.5, config.floatX), name="Woutb")

        #initial vectors
        self.h0 = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="h0")  # Does nothing I know
        self.c0 = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="c0")  # Does nothing I know

        self.params = [self.emb,
                        #Inputs
                        self.Wix, self.Wih, self.Wic, self.Wib,
                        #forget
                        self.Wfx, self.Wfh, self.Wfc, self.Wfb,
                        #output
                        self.Wox, self.Woh, self.Woc, self.Wob,
                        #cell we
                        self.Wcx, self.Wch, self.Wcb,
                        #output
                        self.Wo, self.Woutb]

    def get_outputs_info(self, xs):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((xs.shape[1], self.h_size), config.floatX),# h0
                T.zeros((xs.shape[1], self.h_size), config.floatX),# c0
                None, None]

    def get_hidden_function(self):

        def hidden_function(xt, yt, h_tm1, c_tm1):

            ei = T.dot(xt, self.emb)

            #imput gate
            i = T.nnet.sigmoid(T.dot(ei, self.Wix) + T.dot(h_tm1, self.Wih) + T.dot(c_tm1, self.Wic) + self.Wib)

            #forget gate
            f = T.nnet.sigmoid(T.dot(ei, self.Wfx) + T.dot(h_tm1, self.Wfh) + T.dot(c_tm1, self.Wfc) + self.Wfb)

            #proposed_cell
            ct = T.tanh(T.dot(ei, self.Wcx) + T.dot(h_tm1, self.Wch) + self.Wcb)

            #cell
            ct = f*c_tm1 + i*ct

            #output gate
            og = T.nnet.sigmoid(T.dot(ei, self.Wox) + T.dot(h_tm1, self.Woh) + T.dot(ct, self.Woc) + self.Wob)

            ht = og*T.tanh(og)

            # output
            ot = T.dot(ht, self.Wo)+ self.Woutb
            ot = T.nnet.softmax(ot)

            # loss
            loss = utils.crossEntropy(yt, ot)

            return ht, ct, ot, loss

        return hidden_function



