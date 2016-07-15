import data_loader
import RNN
import ipdb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    v_size = 100
    d = data_loader.data_crawler(maxCount=v_size)
    r = RNN.rnn(h_size = 50, e_size = 50, v_size=v_size, lr=0.1)

    losses = r.train(100, d)
    noS = 0
    print d.switchRep(d[noS])
    pred0 = d[noS][0]
    pred = r.predict(d[noS])[:-1]
    pred = [pred0]+list(pred)
    print d.switchRep(pred)

    print "It's working!!"
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.show()
