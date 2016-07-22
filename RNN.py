import numpy as np
import theano.tensor as T
import theano
from theano import function, shared, config
from theano.sandbox.rng_mrg import MRG_RandomStreams
import ipdb, utils, time, copy, pickle, os


class RNN:
    def __init__(self, h_size = 3, e_size = 2, v_size = 10, lr = 0.01, momentum=0.9, dropout_rate=0.5):

        self.h_size = h_size
        self.e_size = e_size
        self.v_size = v_size

        self.lr = lr
        self.momentum = momentum
        self.dropout_rate = dropout_rate

        np.random.seed(seed=1993)
        self.initParams()

        self.initThenoFunctions()

    def initThenoFunctions(self):
        """
        Compile the theano function needed for the network
        :return:
        """

        self.t_fp, self.t_pred = self.getFunc()
        self.t_generate = self.getGenerateFunction()

    def initParams(self):
        """

        :return: The initial parameters of the RNN
        """

        range_emb = 1/float(2*self.e_size)
        self.emb = shared(np.asarray(np.random.uniform(-range_emb, range_emb, (self.v_size, self.e_size)),
                                     config.floatX), name="Emb")
        self.Wx = shared(np.asarray(np.random.normal(0, 0.01, (self.e_size, self.h_size)), config.floatX), name="Wx")
        self.Wh = shared(np.asarray(np.random.normal(0, 0.01,(self.h_size, self.h_size)), config.floatX), name="Wh")
        self.Wo = shared(np.asarray(np.random.normal(0, 0.01,(self.h_size, self.v_size)), config.floatX), name="Wo")

        #biais
        self.Whb = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Whb")
        self.Wob = shared(np.asarray(np.zeros(self.v_size), config.floatX), name="Wob")

        self.params = [self.emb, self.Wx, self.Wh, self.Wo, self.Whb, self.Wob]

        #momentum

    def get_outputs_info(self, m_size):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((m_size, self.h_size), config.floatX),# h0
                None, None]# output, loss

    def train(self, nb_epoch, trainingSet, validSet, metadata, savingPath=None):
        """
        Train the network for a certain number of epoch.

        :param nb_epoch: number of epoch to train the model
        :param trainingSet: The training data
        :param validSet: The valid data
        :param metadata: The metadata
        :param savingPath: the path where we save the best model

        :return: The losses for each epochs
        """

        trainLosses = []
        validLosses = []
        min_loss = np.inf

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

            if savingPath and loss < min_loss:
                min_loss = loss
                utils.save_everything(savingPath, self, metadata)

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
        """
        Makes sure all the sentences in the minibatch are the same length. Also add an empty word at the beginning.
        Plus make the sentences 1-hot.
        :param minibatch: a list of sentences
        :return: a padded list of sentences.
        """
        max_len = max([len(x) for x in minibatch])
        sentences = []

        #ipdb.set_trace()
        for sentence in minibatch:
            sentence = utils.oneHots(sentence, self.v_size)# one hot representation
            sentence = np.pad(sentence, ((1, max_len-len(sentence)), (0, 0)),
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
            ht = T.dot(et, self.dropMeThat(self.Wx)) + T.dot(h_tm1, self.Wh) + self.Whb
            ht = T.nnet.sigmoid(ht)

            # output
            ot = T.dot(ht, self.Wo) + self.Wob
            ot = T.nnet.softmax(ot)

            # loss
            loss = utils.t_crossEntropy(yt, ot)

            return ht, ot, loss

        return hidden_function

    def dropMeThat(self, weight_matrix):

        srng = MRG_RandomStreams(seed=1993)
        mask = srng.binomial(size=weight_matrix.shape,
                             p=1-self.dropout_rate).astype(config.floatX)

        #mask = T.zeros_like(weight_matrix)

        output = weight_matrix*mask
        #return output
        return output

    def getFunc(self):

        #momentum
        #ipdb.set_trace()
        acc_update = {p.name: shared(np.zeros(p.get_value().shape).astype(config.floatX),
                                     name='{}_acc_grad'.format(p.name)) for p in self.params}

        xs = T.ftensor3("xs")# (no_seq, no_minibatch, no_word)
        ys = T.ftensor3("ys")


        outputs, updates = theano.scan(fn=self.get_hidden_function(),
                                       outputs_info=self.get_outputs_info(xs.shape[1]),
                                       sequences=[xs, ys])

        lossT = outputs[-1]
        oT = outputs[-2]

        sum_lossT = lossT.sum() # Le loss is the sum of all the loss in the sequence
        gParams  = T.grad(sum_lossT, self.params)
        #gParams = [T.max(-1, T.min(1, gp)) for gp in gParams] #clipping the gradient.

        updates_value = [(p, self.lr*gp + acc_update[p.name]*self.momentum) for p, gp in zip(self.params, gParams)]
        updates = [(p, p - uv) for p, uv in updates_value]

        #Update the cumulated update for the momentum
        updates += [(acc_update[p.name], uv) for p, uv in updates_value]

        back_prob = function([xs, ys], sum_lossT, updates=updates) #return the total loss of the minibatch
        prediction = function([xs, ys], [oT, sum_lossT])# return the softmaxes, and the loss for every sentences

        return back_prob, prediction

    def generateRandomSequence(self):
        pass

    def getGenerateFunction(self):
        return None

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

        total_loss = []
        print "Getting the perplexity..."
        import time
        for minibatch in dataset:

            clock = time.clock()

            hot_minibatch = self.hotify_minibatch(minibatch)
            m_xs, m_ys = hot_minibatch[:-1], hot_minibatch[1:]
            m_pred_softmax, _ = self.t_pred(m_xs.astype(config.floatX), m_ys.astype(config.floatX))
            m_pred_softmax = m_pred_softmax.transpose((1, 0, 2))
            print "The time: {}".format(time.clock() - clock)
            clock = time.clock()
            average_losses = [utils.crossEntropy(ys, softmax)/len(sentence)
                              for ys, softmax, sentence in zip(m_ys.transpose((1,0,2)), m_pred_softmax, minibatch)]

            total_loss += average_losses
            print "The time: {}".format(time.clock() - clock)

            print "right now it is: {}".format(np.exp2(np.mean(total_loss)))

        perplexity = np.exp2(np.mean(total_loss))

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
        del params["t_generate"]

        pickle.dump(params, open(path, 'w'))

    def load(self, path):
        params = pickle.load(open(path))

        #ipdb.set_trace()
        # Sorry, little hack to recreate new shared variable
        new_params = []
        for key, value in params.iteritems():

                if key in ['h0', 'c0']: # I'm so, so sorry.
                    continue
                try:
                    params[key] = shared(value.get_value().astype(config.floatX), key)
                    new_params.append(params[key])
                except:
                    pass

        params['params'] = new_params

        self.__dict__.update(params)
        self.initThenoFunctions()





class LSTM(RNN):

    def __init__(self, **params):
        RNN.__init__(self, **params)

    def initParams(self):

        #Embedings
        range_emb = 1/float(2*self.e_size)
        self.emb = shared(np.asarray(np.random.uniform(-range_emb, range_emb, (self.v_size, self.e_size)),
                                     config.floatX), name="Emb")

        #Inputs gate weights
        self.Wix = shared(np.asarray(np.random.normal(0, 0.01, (self.e_size, self.h_size)), config.floatX), name="Wix")
        self.Wih = shared(np.asarray(np.random.normal(0, 0.01, (self.h_size, self.h_size)), config.floatX), name="Wih")
        self.Wic = shared(np.diag(np.random.normal(0, 0.01, (self.h_size))).astype(config.floatX), name = "Wic")
        self.Wib = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wib")

        #forget gates weights
        self.Wfx = shared(np.asarray(np.random.normal(0, 0.01, (self.e_size, self.h_size)), config.floatX), name="Wfx")
        self.Wfh = shared(np.asarray(np.random.normal(0, 0.01, (self.h_size, self.h_size)), config.floatX), name="Wfh")
        self.Wfc = shared(np.diag(np.random.normal(0, 0.01, (self.h_size))).astype(config.floatX), name = "Wfc")
        self.Wfb = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wfb")

        #output gate weights
        self.Wox = shared(np.asarray(np.random.normal(0, 0.01, (self.e_size, self.h_size)), config.floatX), name="Wox")
        self.Woh = shared(np.asarray(np.random.normal(0, 0.01, (self.h_size, self.h_size)), config.floatX), name="Woh")
        self.Woc = shared(np.diag(np.random.normal(0, 0.01, (self.h_size))).astype(config.floatX), name = "Woc")
        self.Wob = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wob")

        #cell weights
        self.Wcx = shared(np.asarray(np.random.normal(0, 0.01, (self.e_size, self.h_size)), config.floatX), name="Wcx")
        self.Wch = shared(np.diag(np.random.normal(0, 0.01, (self.h_size))).astype(config.floatX), name = "Wch")
        self.Wcb = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wcb")

        #output weights
        self.Wo = shared(np.asarray(np.random.normal(0, 0.01, (self.h_size, self.v_size)), config.floatX), name="Wo")
        self.Woutb = shared(np.asarray(np.zeros(self.v_size), config.floatX), name="Woutb")

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

    def get_outputs_info(self, m_size):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((m_size, self.h_size), config.floatX),# h0
                T.zeros((m_size, self.h_size), config.floatX),# c0
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

            ht = og*T.tanh(ct)

            # output
            ot = T.dot(ht, self.Wo)+ self.Woutb
            ot = T.nnet.softmax(ot)

            # loss
            loss = utils.t_crossEntropy(yt, ot)

            return ht, ct, ot, loss

        return hidden_function


    def getGenerateFunction(self):
        # generate a random initial word
        # scannnnn
        # tada

        #matrises
        xt = T.fmatrix('xt')
        ht = T.fmatrix("ht")
        ct = T.fmatrix("ct")

        h_tp1, c_tp1, o_tp1, loss_tp1 = self.get_hidden_function()(xt, T.zeros_like(xt), ht, ct)

        f = function([xt, ht, ct], [h_tp1, c_tp1, o_tp1])
        return f


    def generateRandomSequence(self, maxLength=10):


        xi = np.zeros((1, self.v_size), dtype=config.floatX)  # Our initial word(s)
        hi = np.zeros((1, self.h_size), dtype=config.floatX)
        ci = np.zeros((1, self.h_size), dtype=config.floatX)

        sentence = []
        for i in range(maxLength):
            hi, ci, oi = self.t_generate(xi, hi, ci)

            wordIdx = np.random.choice(a=self.v_size, p=oi[0])
            word = np.zeros_like(xi)
            word[0][wordIdx] = 1

            sentence.append(wordIdx)
            xi = word

        return sentence


