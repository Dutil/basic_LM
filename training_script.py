import data_loader
import RNN
import ipdb
import matplotlib.pyplot as plt
import argparse
import os, pickle

def save_everything(saving_path, rnn, metadata):

    rnn_file_name = os.path.join(saving_path, "rnn.pkl")
    metadata_name = os.path.join(saving_path, "metadata")

    rnn.save(rnn_file_name)
    pickle.dump(metadata, open(metadata_name, 'w'))

def load_everything(loading_path):

    rnn_file_name = os.path.join(loading_path, "rnn.pkl")
    metadata_name = os.path.join(loading_path, "metadata")

    # ATTENTION, LSTM ne marche pas, je sais.
    rnn = RNN.RNN()
    rnn.load(rnn_file_name)

    metadata = pickle.load(open(metadata_name))

    return rnn, metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mSize", type=int, default=5)
    parser.add_argument("--eSize", type=int, default=5)
    parser.add_argument("--nbEpoch", type=int, default=10)
    parser.add_argument("--dataFolder", default="data")
    parser.add_argument("--hSize", type=int, default=50)
    parser.add_argument("--embSize", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.02)

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

    r = RNN.RNN(h_size = h_size, e_size = embSize, v_size=d.nbWords, lr=lr)

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
    save_everything("saving", r, d)
    pickle.dump([trainingLosses, validLosses], open("debug_loss", 'w'))

    #plt.plot(trainingLosses)
    #plt.plot(validLosses)
    #plt.ylabel("Loss")
    #plt.savefig("Losses.png")

    #rnn, metadata = load_everything("saving")
    #print rnn.getLoss(testset)
    #plt.show()
