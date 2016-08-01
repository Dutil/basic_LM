import utils
import argparse
import ipdb
import data_loader


if __name__ == "__main__":

    parser = argparse.parser = argparse.ArgumentParser()

    parser.add_argument("--folder")
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--nb", type=int, default=10)

    args = parser.parse_args()

    folder = args.folder
    length = args.length
    nb = args.nb

    print "loading the RNN..."
    rnn, d = utils.load_everything(folder)

    #d = data_loader.data_crawler(folder=folder, maxCount=10000000)

    #All the datasets
    trainset = data_loader.data_iterator(data=d.all_data[0], e_size=-1,
                                      m_size=128, vocab=d.vocab,
                                      wordMapping=d.wordMapping)

    testset = data_loader.data_iterator(data=d.all_data[2], e_size=-1,
                                         m_size=128, vocab=d.vocab,
                                         wordMapping=d.wordMapping)

    testset = data_loader.predict_noisy_self(testset)


    #print "for trainset"
    #perplexity = rnn.getPerplexity(trainset)
    #print perplexity

    print "for testset"
    perplexity = rnn.getPerplexity(testset)
    print perplexity

    #for i in range(nb):
    #    sentence = rnn.generateRandomSequence(length)
    #    print " ".join(trainset.switchRep(sentence))
