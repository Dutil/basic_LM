import data_loader
import RNN
import ipdb
import matplotlib.pyplot as plt
import argparse
import os, pickle, utils




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mSize", type=int, default=5)
    parser.add_argument("--eSize", type=int, default=-1)
    parser.add_argument("--nbEpoch", type=int, default=10)
    parser.add_argument("--dataFolder", default="data")
    parser.add_argument("--hSize", type=int, default=100)
    parser.add_argument("--embSize", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--savef", default="saving")
    parser.add_argument("--rnnClass", default="RNN")

    args = parser.parse_args()

    # Parameters and stuff
    v_size = 10000000

    m_size = args.mSize # minibatch size
    epoch_size = args.eSize # The number of minibatch in an epoch
    nbEpoch = args.nbEpoch # The number of epoch to do
    folder = args.dataFolder
    h_size = args.hSize
    embSize = args.embSize
    lr = args.lr
    save_folder = args.savef

    rnnClass = args.rnnClass
    data_iterator_type = data_loader.predict_next_iterator
    if rnnClass == "DAE":
        data_iterator_type = data_loader.predict_noisy_self

    rnn_class = RNN.RNN
    exec "rnn_class = RNN.{}".format(rnnClass)

    d = data_loader.data_crawler(folder=folder, maxCount=v_size)

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

    trainset = data_loader.predict_noisy_self(trainset)
    validset = data_loader.predict_noisy_self(validset)
    testset = data_loader.predict_noisy_self(testset)

    r_layer = rnn_class(h_size = h_size, e_size = embSize, v_size=d.nbWords, name="DAE_1")
    mlp = RNN.MLP(v_size=d.nbWords, lr=lr)
    mlp.layers.append(r_layer)

    # Training
    trainingLosses, validLosses = mlp.train(nbEpoch, trainset, validset, d, save_folder)

    #Getting some prediction, for fun.
    noS = 1
    for i, sentence in zip(range(1), trainset):
        #ipdb.set_trace()
        try:
            print trainset.iterator.switchRep(sentence[0][:,0,:])
            pred = mlp.predict(sentence)[0]
            print trainset.iterator.switchRep(pred)
        except:
            ipdb.set_trace()

    mlp2 = RNN.MLP(v_size=d.nbWords, lr=lr)
    mlp2.load("saving/rnn.pkl")
    ipdb.set_trace()
    testPer = mlp.getPerplexity(testset)
    testLoss = mlp.getLoss(testset)
    print "The perplexity is: {}, the loss is: {}".format(testPer, testLoss)

    # Showing the loss, for fun.

    #utils.save_everything(save_folder, mlp, d)

    pickle.dump([trainingLosses, validLosses],
                open(os.path.join(save_folder,"debug_loss"), 'w'))


    print "It's working!!"
    #plt.plot(trainingLosses)
    #plt.plot(validLosses)
    #plt.ylabel("Loss")
    #plt.savefig("Losses.png")

    #rnn, metadata = load_everything("saving")
    #print rnn.getLoss(testset)
    #plt.show()
