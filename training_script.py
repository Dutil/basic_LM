import data_loader
import RNN
import ipdb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    v_size = 100
    m_size = 2 # minibatch size
    d = data_loader.data_crawler(maxCount=v_size, m_size=m_size)
    r = RNN.RNN(h_size = 50, e_size = 50, v_size=d.metadata['nbWords'], lr=0.02, m_size=m_size)

    losses = r.train(100, d)
    noS = 1
    print d.switchRep(d[noS])
    pred0 = d[noS][0]
    pred = r.predict([d[noS]])[0][:-1]
    pred = [pred0]+list(pred)
    print d.switchRep(pred)

    p = r.get_perplexity(d)
    print "The perplexity is: {}".format(p)

    print "It's working!!"
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.show()
