import utils
import argparse
import ipdb
import data_loader


if __name__ == "__main__":

    parser = argparse.parser = argparse.ArgumentParser()

    parser.add_argument("--folder")
    parser.add_argument("--p", type=float, default=0.0)
    parser.add_argument("--nb", type=int, default=10)

    args = parser.parse_args()

    folder = args.folder
    p = args.p
    nb = args.nb

    print "loading the RNN..."
    rnn, d = utils.load_everything(folder)

    #d = data_loader.data_crawler(folder=folder, maxCount=10000000)

    #All the datasets
    testset = data_loader.data_iterator(data=d.all_data[2], e_size=-1,
                                         m_size=1, vocab=d.vocab,
                                         wordMapping=d.wordMapping)

    testset = data_loader.predict_noisy_self(testset, p)

    for i, sentence in zip(range(nb), testset):

        #ipdb.set_trace()
        print testset.iterator.switchRep(sentence[0][:,0,:])
        print testset.iterator.switchRep(rnn.predict(sentence)[0])