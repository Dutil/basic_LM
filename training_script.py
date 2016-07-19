import data_loader
import RNN
import ipdb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Parameters and stuff
    v_size = 10000000
    m_size = 10 # minibatch size
    epoch_size = 1 # The number of minibatch in an epoch
    nbEpoch = 1

    d = data_loader.data_crawler(folder="data", maxCount=v_size)

    #All the datasets
    trainset = data_loader.data_iterator(data=d.all_data[0], e_size=epoch_size,
                                      m_size=m_size, vocab=d.vocab,
                                      wordMapping=d.wordMapping)

    validset = data_loader.data_iterator(data=d.all_data[1], e_size=epoch_size,
                                         m_size=m_size, vocab=d.vocab,
                                         wordMapping=d.wordMapping)

    testset = data_loader.data_iterator(data=d.all_data[2], e_size=epoch_size,
                                         m_size=m_size, vocab=d.vocab,
                                         wordMapping=d.wordMapping)

    r = RNN.RNN(h_size = 50, e_size = 50, v_size=d.nbWords, lr=0.02, m_size=m_size)

    # Training
    trainingLosses, validLosses = r.train(nbEpoch, trainset, validset)

    #Getting some prediction, for fun.
    noS = 1
    print trainset.switchRep(trainset[noS])
    pred0 = trainset[noS][0]
    pred = r.predict([trainset[noS]])[0][:-1]
    pred = [pred0]+list(pred)
    print trainset.switchRep(pred)

    testPer = r.getPerplexity(testset)
    testLoss = r.getLoss(testset)
    print "The perplexity is: {}, the loss is: {}".format(testPer, testLoss)

    # Showing the loss, for fun.
    print "It's working!!"
    plt.plot(trainingLosses)
    plt.plot(validLosses)
    plt.ylabel("Loss")
    plt.savefig("Losses.png")
    #plt.show()
