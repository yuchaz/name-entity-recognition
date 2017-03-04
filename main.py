from train import FeatureRichModel, Perceptron
from Corpora import Corpus

learning_rate = 1

def main():
    train_corpus = Corpus.trainCorpus()
    dev_corpus = Corpus.devCorpus()
    feature_model = FeatureRichModel(corpus=train_corpus)
    perceptron = Perceptron(frm=feature_model, eta=learning_rate).fit(train_corpus)
    ner_tags = perceptron.predict(dev_corpus)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    try:
        import time
        start = time.time()
        print start
        main()
        print time.time()-start
    except:
        import sys, pdb, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
