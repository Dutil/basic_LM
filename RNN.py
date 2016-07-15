import numpy as np
import theano.tensor as T
import theano
from theano import function, shared, config
import ipdb
import utils



class rnn:
    def __init__(self, h_size = 3, e_size = 2, v_size = 10, lr = 0.01):

        self.h_size = h_size
        self.e_size = e_size
        self.v_size = v_size

        self.lr = lr

        np.random.seed(seed=1993)
        self.initParams()

        self.t_fp, self.t_pred = self.getFunc()

    def initParams(self):
        self.emb = shared(np.asarray(np.random.rand(self.v_size, self.e_size) - 0.5, config.floatX), name="Emb")
        self.Wx = shared(np.asarray(np.random.rand(self.e_size, self.h_size) - 0.5, config.floatX), name="Wx")
        self.Wh = shared(np.asarray(np.random.rand(self.h_size, self.h_size) - 0.5, config.floatX), name="Wh")
        self.Wo = shared(np.asarray(np.random.rand(self.h_size, self.v_size) - 0.5, config.floatX), name="Wo")
        self.h0 = shared(np.asarray(np.zeros(self.h_size), config.floatX), name="h0")# Does nothing I know

        self.outputs_info=[self.h0, None, None] #h0, o, loss
        self.params = [self.emb, self.Wx, self.Wh, self.Wo]

    def train(self, nb_epoch, data):
        losses = []
        for i in range(nb_epoch):
            print "doing epoch {}".format(i)
            loss = self.doOneEpoch(data)
            losses.append(loss)

        return losses

    def doOneEpoch(self, data):

        losses = []
        for sentence in data:
            sentence = utils.oneHots(sentence, self.v_size)
            loss = self.forwardPass(sentence)
            losses.append(loss)
        return sum(losses)

    def forwardPass(self, sentence):
        _, _, loss = self.t_fp(sentence[:-1], sentence[1:])
        return loss

    def get_hidden_function(self):

        def hidden_function(xi, yi, h_tm1):

            ei = T.dot(xi, self.emb)

            # hidden layer
            hi = T.dot(ei, self.Wx) + T.dot(h_tm1, self.Wh)
            hi = T.nnet.sigmoid(hi)

            # output
            oi = T.dot(hi, self.Wo)
            oi = T.nnet.softmax(oi)

            # loss
            loss = utils.crossEntropy(yi, oi)

            return hi, oi, loss

        return hidden_function

    def getFunc(self):

        xs = T.fmatrix("xs")
        ys = T.fmatrix("ys")

        [hT, oT, lossT], updates = theano.scan(fn=self.get_hidden_function(),
                                          outputs_info=self.outputs_info,
                                          sequences=[xs, ys])

        lossT = lossT.sum()
        gParams  = T.grad(lossT, self.params)
        updates = [(p, p - self.lr*gp) for p, gp in zip(self.params, gParams)]

        back_prob = function([xs, ys], [hT, oT, lossT], updates=updates)
        prediction = function([xs, ys], oT)
        return back_prob, prediction

    def predict(self, sentence):

        sentence = utils.oneHots(sentence, self.v_size)
        pred_sm = self.t_pred(sentence, sentence[::-1])

        pred = np.zeros(len(sentence)).astype(np.int32)
        for i in range(len(pred_sm)):
            pred[i] = np.argmax(pred_sm[i])
        return pred
