from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack, csr_matrix
from itertools import product
from collections import defaultdict
from train.viterbi_decode import viterbi_decode as viterbi_dc
import numpy as np

START_SYMBOL = '<START>'
STOP_SYMBOL = '<STOP>'


class FeatureRichModel(object):
    def __init__(self, corpus):
        self.token_set, self.pos_set, self.chunk_set = corpus.get_sets()
        self.token_set.add(STOP_SYMBOL)
        NER_tag_choices = 'ORG PER LOC MISC'.split()
        BIO_choices = 'B I'.split()
        self.ner_tags_set = set(["{}-{}".format(tag[0], tag[1]) for tag in product(BIO_choices, NER_tag_choices)])
        self.ner_tags_set.update([START_SYMBOL, STOP_SYMBOL, 'O'])
        self.token_ner_encoder = OneHotEncoder([token_vs_ner(*token_ner)
            for token_ner in product(self.token_set, self.ner_tags_set)])
        self.feature_dim = 2*(self.token_ner_encoder.feature_size)
        self.ner_tags_set.difference_update([START_SYMBOL, STOP_SYMBOL])
        self.ner_tags_set = list(self.ner_tags_set)

    def local_feature_trans(self, sentence, current_ner_tag, prev_ner_tag, k):
        current_token = STOP_SYMBOL if k == sentence.length else sentence.tokens[k]
        token_curr_ner = self.token_ner_encoder.transform(token_vs_ner(current_token, current_ner_tag))
        token_prev_ner = self.token_ner_encoder.transform(token_vs_ner(current_token, prev_ner_tag))
        return hstack([token_curr_ner, token_prev_ner], format='csr')

    def global_feature_trans(self, sentence, ner_tags):
        global_feature = csr_matrix((1,self.feature_dim))
        for idx, ner_tag in enumerate(ner_tags):
            prev_ner_tag = START_SYMBOL if idx == 0 else ner_tags[idx-1]
            global_feature += self.local_feature_trans(sentence, ner_tag, prev_ner_tag, idx)
        return global_feature

    def score(self, w_weight_vector, sentence, current_ner_tag, prev_ner_tag, k):
        return w_weight_vector.dot(self.local_feature_trans(
            sentence, current_ner_tag, prev_ner_tag, k).transpose()).toarray()[0,0]

    def generate_init_score(self,w_weight_vector,sentence):
        init_score = [self.score(w_weight_vector,sentence,ner_tag,START_SYMBOL,0) for ner_tag in self.ner_tags_set]
        return np.array(init_score)

    def generate_trans_score(self,w_weight_vector,sentence):
        trans_score = [[self.score(w_weight_vector,sentence,ner_tag,prev_ner_tag,0)
            for ner_tag in self.ner_tags_set] for prev_ner_tag in self.ner_tags_set]
        return np.array(trans_score)
    def viterbi_decode(self,sentence,init_score,trans_score):
        decode = viterbi_dc(sentence,init_score,trans_score)
        return map(lambda k: self.ner_tags_set[k], decode)
def token_vs_ner(token, ner):
    return "{}_vs_{}".format(token, ner)

class OneHotEncoder(object):
    def __init__(self, list_to_encode):
        self.feature_dict = dict(zip(list_to_encode, range(len(list_to_encode))))
        self.feature_size = len(list_to_encode)
    def transform(self, token):
        idx = self.feature_dict.get(token, -1)
        if not idx == -1:
            row = [0]
            col = [idx]
            data = [1.]
            return csr_matrix( (data, (row, col)), shape=(1,self.feature_size))
