from train import FeatureRichModel, Perceptron
from Corpora import Corpus
import argparse

parser = argparse.ArgumentParser(description="Run Name Entity Recognition powered by Feature Rich Model")
parser.add_argument('--lrn-rate', '-l', type=float, default=1,
    help="The learning rate eta of the Perceptron, should be a float.")
parser.add_argument('--epoch', '-e', type=int, default=5,
    help="The epoch time of the Perceptron, should be an int")
parser.add_argument('--no-c-token', nargs='?', default=False, const=True,
    help="Specify when do not need current token v.s. NER tag feature")
parser.add_argument('--no-p-token', nargs='?', default=False, const=True,
    help="Specify when do not need previous token v.s. NER tag feature")
parser.add_argument('--no-c-pos', nargs='?', default=False, const=True,
    help="Specify when do not need current pos v.s. NER tag feature")
parser.add_argument('--no-p-pos', nargs='?', default=False, const=True,
    help="Specify when do not need previous pos v.s. NER tag feature")
parser.add_argument('--no-c-chunk', nargs='?', default=False, const=True,
    help="Specify when do not need current chunk v.s. NER tag feature")
parser.add_argument('--no-p-chunk', nargs='?', default=False, const=True,
    help="Specify when do not need previous chunk v.s. NER tag feature")

args = parser.parse_args()
eta = args.lrn_rate
epoch = args.epoch

def main():
    train_corpus = Corpus.trainCorpus()
    dev_corpus = Corpus.devCorpus()
    feature_model = FeatureRichModel(corpus=train_corpus)
    perceptron = Perceptron(frm=feature_model, eta=eta, epoch=epoch).fit(train_corpus)
    ner_tags = perceptron.predict(dev_corpus)
    dev_corpus.save('./output_{}_eta_{}_epoch_{}.txt'.format(
        "dev", eta, epoch))

if __name__ == '__main__':
    try:
        import time
        start = time.time()
        main()
        print time.time()-start
    except:
        import sys, pdb, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
