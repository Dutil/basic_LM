import utils
import argparse
import ipdb
import data_loader


if __name__ == "__main__":

    parser = argparse.parser = argparse.ArgumentParser()

    parser.add_argument("--folder")
    parser.add_argument("--LSTM", action="store_true")
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--nb", type=int, default=10)

    args = parser.parse_args()

    folder = args.folder
    is_LSTM = args.LSTM
    length = args.length
    nb = args.nb

    print "loading the RNN..."
    rnn, d = utils.load_everything(folder, is_LSTM)

    #d = data_loader.data_crawler(folder=folder, maxCount=10000000)

    #All the datasets
    trainset = data_loader.data_iterator(data=d.all_data[0], e_size=-1,
                                      m_size=128, vocab=d.vocab,
                                      wordMapping=d.wordMapping)

    for i in range(nb):
        sentence = rnn.generateRandomSequence(length)
        print " ".join(trainset.switchRep(sentence))


    #random_s = rnn.generateRandomSequence()

    #noS = 1
    #print trainset.switchRep(trainset[noS])
    #pred = rnn.predict([trainset[noS]])[0]
    #pred = list(pred)
    #print trainset.switchRep(pred)

    #perplexity = rnn.getPerplexity(testset)
    #print "the perplexity on the testset is: {}".format(perplexity)

    #ipdb.set_trace()
