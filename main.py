from train import FeatureRichModel, Perceptron
from Corpora import Corpus
import argparse

parser = argparse.ArgumentParser(description="Run Name Entity Recognition powered by Feature Rich Model")
parser.add_argument('--lrn-rate', '-l', type=float, default=1,
    help="The learning rate eta of the Perceptron, should be a float.")
parser.add_argument('--epoch', '-e', type=int, default=5,
    help="The epoch time of the Perceptron, should be an int")

parser.add_argument('--no-ctoken', nargs='?', default=True, const=False,
    help="Specify when do not need current token v.s. NER tag feature")
parser.add_argument('--no-ptoken', nargs='?', default=True, const=False,
    help="Specify when do not need previous token v.s. NER tag feature")
parser.add_argument('--no-ftoken', nargs='?', default=True, const=False,
    help="Specify when do not need future token v.s. NER tag feature")
parser.add_argument('--no-cpos', nargs='?', default=True, const=False,
    help="Specify when do not need current pos v.s. NER tag feature")
parser.add_argument('--no-ppos', nargs='?', default=True, const=False,
    help="Specify when do not need previous pos v.s. NER tag feature")
parser.add_argument('--no-fpos', nargs='?', default=True, const=False,
    help="Specify when do not need future pos v.s. NER tag feature")
parser.add_argument('--no-cchunk', nargs='?', default=True, const=False,
    help="Specify when do not need current chunk v.s. NER tag feature")
parser.add_argument('--no-pchunk', nargs='?', default=True, const=False,
    help="Specify when do not need previous chunk v.s. NER tag feature")
parser.add_argument('--no-fchunk', nargs='?', default=True, const=False,
    help="Specify when do not need future chunk v.s. NER tag feature")

parser.add_argument('--test', '-T', nargs='?', default=False, const=True,
    help="Specify when switch to test mode.")
parser.add_argument('--small', '-S', nargs='?', default=False, const=True,
    help="Specify when use small corpus.")

args = parser.parse_args()
eta = args.lrn_rate
epoch = args.epoch
is_test = args.test
is_small = args.small
feature_dict = {
    "if_current_token": args.no_ctoken, "if_prev_token": args.no_ptoken, "if_future_token": args.no_ftoken,
    "if_current_pos": args.no_cpos, "if_prev_pos": args.no_ppos, "if_future_token": args.no_fpos,
    "if_current_chunk": args.no_cchunk, "if_prev_chunk": args.no_pchunk, "if_future_chunk": args.no_fchunk
}

def main():
    train_corpus = Corpus.trainCorpus(is_small=is_small) \
        if not is_test else Corpus.combinedCorpus('train','dev', is_small=is_small)
    eval_corpus = Corpus.devCorpus(is_small=is_small) \
        if not is_test else Corpus.testCorpus(is_small=is_small)
    eval_name = 'dev' if not is_test else 'test'
    if is_small: eval_name = "small_{}".format(eval_name)
    feature_model = FeatureRichModel(corpus=train_corpus, **feature_dict)
    perceptron = Perceptron(frm=feature_model, eta=eta, epoch=epoch).fit(train_corpus)
    ner_tags = perceptron.predict(eval_corpus)
    feature_code = "{}{}{}{}{}{}{}{}{}".format(
        int(feature_dict["if_current_token"]),int(feature_dict["if_prev_token"]),int(feature_dict["if_future_token"]),
        int(feature_dict["if_current_pos"]),int(feature_dict["if_prev_pos"]), int(feature_dict["if_future_pos"]),
        int(feature_dict["if_current_chunk"]),int(feature_dict["if_prev_chunk"]), int(feature_dict["if_future_chunk"]))
    eval_corpus.save('./output/{}_eta_{}_epoch_{}_{}.txt'.format(
        eval_name, eta, epoch, feature_code))

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
