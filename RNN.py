import numpy as np
import theano.tensor as T
import theano
from theano import function, shared, config
from theano.sandbox.rng_mrg import MRG_RandomStreams
import ipdb, utils, time, copy, pickle, os




class MLP:

    def __init__(self, v_size = 10, lr = 0.01, momentum=0.0):


        self.v_size = v_size

        self.lr = lr
        self.momentum = momentum

        np.random.seed(seed=1993)

        self.layers=[]


    def initThenoFunctions(self):
        """
        Compile the theano function needed for the network
        :return:
        """

        self.t_fp, self.t_pred = self.getFunc()
        #self.t_generate = self.getGenerateFunction()

    def getParams(self):

        params = {}

        for layer in self.layers:
            for p in layer.getParams():
                params[p.name] = p

        return params



    def getFunc(self):

        acc_update = {p.name: shared(np.zeros(p.get_value().shape).astype(config.floatX),
                                     name='{}_acc_grad'.format(p.name)) for p in self.getParams().values()}

        xs = T.ftensor3("xs")  # (no_seq, no_minibatch, no_word)
        ys = T.ftensor3("ys")

        inputs = xs
        outputs = None
        for layer in self.layers:
            outputs = layer.fprop(inputs, ys)
            inputs = outputs[0] # Correspond to the hidden state

        lossT = outputs[-1]
        oT = outputs[-2]

        sum_lossT = lossT.sum()  # Le loss is the sum of all the loss in the sequence
        gParams = T.grad(sum_lossT, self.getParams().values())
        # gParams = [T.max(-1, T.min(1, gp)) for gp in gParams] #clipping the gradient.

        updates_value = [(p, self.lr * gp + acc_update[p.name] * self.momentum)
                         for p, gp in zip(self.getParams().values(), gParams)]
        updates = [(p, p - uv) for p, uv in updates_value]

        # Update the cumulated update for the momentum
        updates += [(acc_update[p.name], uv) for p, uv in updates_value]

        back_prob = function([xs, ys], sum_lossT, updates=updates)  # return the total loss of the minibatch
        prediction = function([xs, ys], [oT, sum_lossT])  # return the softmaxes, and the loss for every sentences

        return back_prob, prediction


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
        self.initThenoFunctions()

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
        # ipdb.set_trace()
        for minibatch in data:
            #sentences = self.hotify_minibatch(minibatch)

            loss = self.forwardPass(minibatch)
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

        # ipdb.set_trace()
        for sentence in minibatch:
            sentence = utils.oneHots(sentence, self.v_size)  # one hot representation
            sentence = np.pad(sentence, ((1, max_len - len(sentence)), (0, 0)),
                              'constant', constant_values=(0))  # padding to the max length
            sentences.append(sentence)

        sentences = np.array(sentences).astype(config.floatX)
        sentences = sentences.transpose((1, 0, 2))
        return sentences


    def forwardPass(self, minibatch):
        loss = self.t_fp(*minibatch) #minibatch[:-1], minibatch[1:])
        return loss

    def getLoss(self, dataset):
        """
        Get the total loss of a particular dataset

        :param dataset:
        :return:
        """

        totalLoss = 0.0
        for minibatch in dataset:
            #hot_minibatch = self.hotify_minibatch(minibatch)
            m_xs, m_ys = minibatch # hot_minibatch[:-1], hot_minibatch[1:]
            _, loss = self.t_pred(m_xs, m_ys)
            totalLoss += loss

        return totalLoss


    def predict(self, minibatch):
        """
        For every sentence, for every word xi, predict the word xi+1
        :param sentences: the list of sentences
        :return:
        """

        pred_softmax, _ = self.t_pred(*minibatch)  # the softmaxes
        pred_softmax = pred_softmax.transpose((1, 0, 2))
        preds = []

        # Get the word with the maximum probability for each sentences
        for pred_sentence in pred_softmax:

            pred = np.zeros(len(pred_sentence)).astype(np.int32)
            for i in range(len(pred_sentence)):
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

            m_xs, m_ys = minibatch
            m_pred_softmax, _ = self.t_pred(m_xs.astype(config.floatX), m_ys.astype(config.floatX))
            m_pred_softmax = m_pred_softmax.transpose((1, 0, 2))
            print "The time: {}".format(time.clock() - clock)
            clock = time.clock()
            average_losses = [utils.crossEntropy(ys, softmax) / len(sentence)
                              for ys, softmax, sentence in zip(m_ys.transpose((1, 0, 2)), m_pred_softmax, minibatch)]

            total_loss += average_losses
            print "The time: {}".format(time.clock() - clock)

            print "right now it is: {}".format(np.exp2(np.mean(total_loss)))

        perplexity = np.exp2(np.mean(total_loss))

        return perplexity

    def save(self, path):
        # I know, I know. I have some diplicate, but I don't really care :)
        to_save = {'layers':{}}

        for layer in self.layers:
            params = layer.getParamsValues()
            to_save['layers'][layer.name] = {}
            to_save['layers'][layer.name]['params'] = params
            to_save['layers'][layer.name]['class'] = layer.__class__

        to_save['v_size'] = self.v_size
        to_save['lr'] = self.lr
        to_save['momemtum'] = self.momentum

        #I don't really care about saving the functions.
        pickle.dump(to_save, open(path, 'w'))

    def load(self, path):

        print "init of the layer..."
        to_load = pickle.load(open(path))

        self.v_size = to_load['v_size']
        self.lr = to_load['lr']
        self.momentum = to_load['momemtum']

        self.layers = []
        for layer_name, layer_info in to_load['layers'].iteritems():
            class_name = layer_info['class']
            params = layer_info['params']
            layer = class_name()
            layer.name = layer_name
            layer.loadPrams(params)

            self.layers.append(layer)

        print "Compiling theano..."
        self.initThenoFunctions()
        print "Done"

class RNN(object):
    def __init__(self, h_size = 3, e_size = 2, v_size = 10, dropout_rate=0.5, name="RNN_layer_1"):

        self.h_size = h_size
        self.e_size = e_size
        self.v_size = v_size

        self.dropout_rate = dropout_rate
        self.name=name

        np.random.seed(seed=1993)
        self.initParams()

        #self.initThenoFunctions()

    def initParams(self):
        """

        :return: The initial parameters of the RNN
        """

        range_emb = 1/float(2*self.e_size)
        self.Emb = shared(np.asarray(np.random.uniform(-range_emb, range_emb, (self.v_size, self.e_size)),
                                     config.floatX), name="Emb")
        self.Wx = shared(np.asarray(np.random.normal(0, 0.1, (self.e_size, self.h_size)), config.floatX), name="Wx")
        self.Wh = shared(np.asarray(np.random.normal(0, 0.1,(self.h_size, self.h_size)), config.floatX), name="Wh")
        self.Wo = shared(np.asarray(np.random.normal(0, 0.1,(self.h_size, self.v_size)), config.floatX), name="Wo")

        #biais
        self.Whb = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Whb")
        self.Wob = shared(np.asarray(np.zeros(self.v_size), config.floatX), name="Wob")


    def get_outputs_info(self, m_size):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((m_size, self.h_size), config.floatX),# h0
                None, None]# output, loss

    def fprop(self, xs, ys):

        outputs, updates = theano.scan(fn=self.get_hidden_function(),
                                       outputs_info=self.get_outputs_info(xs.shape[1]),
                                       sequences=[xs, ys])

        return outputs


    def get_hidden_function(self):

        def hidden_function(xt, yt, h_tm1):

            et = T.dot(xt, self.dropMeThat(self.Emb))

            # hidden layer
            ht = T.dot(et, self.dropMeThat(self.Wx)) + T.dot(h_tm1, self.Wh) + self.Whb
            ht = T.nnet.sigmoid(ht)

            # output
            ot = T.dot(ht, self.dropMeThat(self.Wo)) + self.Wob
            ot = T.nnet.softmax(ot)

            # loss
            loss = utils.t_crossEntropy(yt, ot)

            return ht, ot, loss

        return hidden_function

    def dropMeThat(self, weight_matrix):

        srng = MRG_RandomStreams(np.random.randint(100000))
        mask = srng.binomial(size=weight_matrix.shape,
                             p=1-self.dropout_rate).astype(config.floatX)

        #mask = T.zeros_like(weight_matrix)

        output = weight_matrix*mask
        #return output
        return output

    def generateRandomSequence(self):
        pass

    def getGenerateFunction(self):
        return None

    def getParams(self):
        return [self.Emb, self.Wx, self.Wh, self.Wo, self.Whb, self.Wob]

    def getParamsValues(self):

        outputs = {}

        outputs['params'] = {p.name: p.get_value() for p in self.getParams()}
        outputs['h_size'] = self.h_size
        outputs['e_size'] = self.e_size
        outputs['v_size'] = self.v_size


        return outputs

    def loadPrams(self, params):


        self.h_size = params['h_size']
        self.e_size = params['e_size']
        self.v_size = params['v_size']

        for key, value in params['params'].iteritems():
            self.__dict__[key] = shared(value.astype(config.floatX), name=key)

class LSTM(RNN):

    def __init__(self, **params):
        RNN.__init__(self, **params)

    def initParams(self):

        #Embedings
        range_emb = 1/float(2*self.e_size)
        self.Emb = shared(np.asarray(np.random.uniform(-range_emb, range_emb, (self.v_size, self.e_size)),
                                     config.floatX), name="Emb")

        #Inputs gate weights
        self.Wix = shared(np.asarray(np.random.normal(0, 0.1, (self.e_size, self.h_size)), config.floatX), name="Wix")
        self.Wih = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Wih")
        self.Wic = shared(np.diag(np.random.normal(0, 0.1, (self.h_size))).astype(config.floatX), name = "Wic")
        self.Wib = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wib")

        #forget gates weights
        self.Wfx = shared(np.asarray(np.random.normal(0, 0.1, (self.e_size, self.h_size)), config.floatX), name="Wfx")
        self.Wfh = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Wfh")
        self.Wfc = shared(np.diag(np.random.normal(0, 0.1, (self.h_size))).astype(config.floatX), name = "Wfc")
        self.Wfb = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wfb")

        #output gate weights
        self.Wox = shared(np.asarray(np.random.normal(0, 0.1, (self.e_size, self.h_size)), config.floatX), name="Wox")
        self.Woh = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Woh")
        self.Woc = shared(np.diag(np.random.normal(0, 0.1, (self.h_size))).astype(config.floatX), name = "Woc")
        self.Wob = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wob")

        #cell weights
        self.Wcx = shared(np.asarray(np.random.normal(0, 0.1, (self.e_size, self.h_size)), config.floatX), name="Wcx")
        self.Wch = shared(np.diag(np.random.normal(0, 0.1, (self.h_size))).astype(config.floatX), name = "Wch")
        self.Wcb = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="Wcb")

        #output weights
        self.Wo = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.v_size)), config.floatX), name="Wo")
        self.Woutb = shared(np.asarray(np.zeros(self.v_size), config.floatX), name="Woutb")

    def get_outputs_info(self, m_size):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((m_size, self.h_size), config.floatX),# h0
                T.zeros((m_size, self.h_size), config.floatX),# c0
                None, None]

    def getParams(self):
        return [self.Emb,
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

    def get_hidden_function(self):

        def hidden_function(xt, yt, h_tm1, c_tm1):

            #emb = self.dropMeThat(self.Emb)
            ei = T.dot(xt, self.Emb)

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

class Decoder_LSTM(LSTM):


    def __init__(self, **params):
        LSTM.__init__(self, **params)

    def initParams(self):
        super(Decoder_LSTM, self).initParams()

        #Adding the context weight
        self.Wicon = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Wicon")
        self.Wfcon = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Wfcon")
        self.Wocon = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Wocon")
        self.Wccon = shared(np.asarray(np.random.normal(0, 0.1, (self.h_size, self.h_size)), config.floatX), name="Wccon")


    #Output info, Avoir Les x.

    def getParams(self):

        to_return = super(Decoder_LSTM, self).getParams()
        return to_return + [self.Wicon, self.Wfcon, self.Wocon, self.Wccon]

    def get_outputs_info(self, m_size):
        """
        Return the ouputs_info for the theano.scan function
        :param xs: the sequence over wich the scan will pass
        :return: the outputs_info
        """

        return [T.zeros((m_size, self.h_size), config.floatX),# h0
                T.zeros((m_size, self.h_size), config.floatX),# c0
                T.zeros((m_size, self.v_size), config.floatX),
                None, None]

    def get_hidden_function(self):

        def hidden_function(yt, h_tm1, c_tm1, y_tm1, con):

            emb = self.Emb
            ei = T.dot(y_tm1, emb)

            #imput gate
            i = T.nnet.sigmoid(T.dot(ei, self.Wix) + T.dot(h_tm1, self.Wih) + T.dot(c_tm1, self.Wic)
                               + T.dot(con, self.Wicon) + self.Wib)

            #forget gate
            f = T.nnet.sigmoid(T.dot(ei, self.Wfx) + T.dot(h_tm1, self.Wfh) + T.dot(c_tm1, self.Wfc)
                               + T.dot(con, self.Wfcon) + self.Wfb)

            #proposed_cell
            ct = T.tanh(T.dot(ei, self.Wcx) + T.dot(h_tm1, self.Wch) + T.dot(con, self.Wccon) + self.Wcb)

            #cell
            ct = f*c_tm1 + i*ct

            #output gate
            og = T.nnet.sigmoid(T.dot(ei, self.Wox) + T.dot(h_tm1, self.Woh) + T.dot(ct, self.Woc)
                                + T.dot(con, self.Wocon) + self.Wob)

            ht = og*T.tanh(ct)

            # output
            ot = T.dot(ht, self.Wo)+ self.Woutb
            ot = T.nnet.softmax(ot)
            next_word = T.zeros_like(y_tm1)
            next_word = T.set_subtensor(next_word[:,T.argmax(ot, axis=1)], 1)

            # loss
            loss = utils.t_crossEntropy(yt, ot)

            return ht, ct, next_word, ot, loss

        return hidden_function

class DAE():

    def __init__(self, h_size=10, e_size=10, v_size=10, name="DAE_1"):

        self.h_size = h_size
        self.e_size = e_size
        self.v_size = v_size
        self.name = name
        self.initParams()

    def initParams(self):

        range_emb = 1 / float(2 * self.e_size)
        self.Emb = shared(np.asarray(np.random.uniform(-range_emb, range_emb, (self.v_size, self.e_size)),
                                     config.floatX), name="Emb")

        self.Encoder = LSTM(h_size=self.h_size, e_size=self.e_size, v_size=self.v_size, name="Encoder")
        self.Decoder = Decoder_LSTM(h_size=self.h_size, e_size=self.e_size, v_size=self.v_size, name="Decoder")

        #They both use the same Embeddings
        self.Encoder.Emb = self.Emb
        self.Decoder.Emd = self.Emb

    def fprop(self, noisy_xs, xs):

        #First pass
        outputs, updates = theano.scan(fn=self.Encoder.get_hidden_function(),
                                       outputs_info=self.Encoder.get_outputs_info(xs.shape[1]),
                                       sequences=[noisy_xs, T.zeros_like(noisy_xs)])


        last_hidden_layer = outputs[0][-1]

        #Decoder!
        outputs, updates = theano.scan(fn=self.Decoder.get_hidden_function(),
                                       outputs_info=self.Decoder.get_outputs_info(xs.shape[1]),
                                       sequences=[xs],
                                       non_sequences=last_hidden_layer)

        return outputs

    def getParams(self):
        #ipdb.set_trace()
        return self.Encoder.getParams() + self.Decoder.getParams()

    def getParamsValues(self):

        outputs = {}

        outputs['params'] = {}
        outputs['params']['Encoder'] = self.Encoder.getParamsValues()
        outputs['params']['Decoder'] = self.Decoder.getParamsValues()
        outputs['params']['Emb'] = self.Emb.get_value()

        outputs['h_size'] = self.h_size
        outputs['e_size'] = self.e_size
        outputs['v_size'] = self.v_size

        return outputs


    def loadPrams(self, params):

        self.h_size = params['h_size']
        self.e_size = params['e_size']
        self.v_size = params['v_size']
        self.Emb = shared(params['params']['Emb'].astype(config.floatX), name='Emb')

        self.Encoder.loadPrams(params['params']['Encoder'])
        self.Decoder.loadPrams(params['params']['Decoder'])

        self.Encoder.Emb = self.Emb
        self.Decoder.Emb = self.Emb


class DropOutLayer:

    def __init__(self, dropout_rate=0.0):
        self.dropout_rate=dropout_rate

    def get_matrix(self, weight_matrix):
        srng = MRG_RandomStreams(np.random.randint(100000))
        mask = srng.binomial(size=weight_matrix.shape,
                             p=1 - self.dropout_rate).astype(config.floatX)

        # mask = T.zeros_like(weight_matrix)

        output = weight_matrix * mask
        # return output
        return output