from itertools import product
from collections import defaultdict
from train.viterbi_decode import viterbi_decode as viterbi_dc
import numpy as np

START_SYMBOL = '<START>'
STOP_SYMBOL = '<STOP>'

class FeatureRichModel(object):
    def __init__(self, corpus, **kwargs):
        self.token_set, self.pos_set, self.chunk_set = corpus.get_sets()
        self.token_set.add(STOP_SYMBOL)
        self.pos_set.add(STOP_SYMBOL)
        self.chunk_set.add(STOP_SYMBOL)
        NER_tag_choices = 'ORG PER LOC MISC'.split()
        BIO_choices = 'B I'.split()
        self.ner_tags_set = set(["{}-{}".format(tag[0], tag[1]) for tag in product(BIO_choices, NER_tag_choices)])
        self.ner_tags_set.update([START_SYMBOL, STOP_SYMBOL, 'O'])

        self.token_ner_encoder = OneHotEncoder([sequence_ner_permutizer(*token_ner)
            for token_ner in product(self.token_set, self.ner_tags_set)])
        self.pos_ner_encoder = OneHotEncoder([sequence_ner_permutizer(*pos_ner)
            for pos_ner in product(self.pos_set, self.ner_tags_set)])
        self.chunk_ner_encoder = OneHotEncoder([sequence_ner_permutizer(*chunk_ner)
            for chunk_ner in product(self.chunk_set, self.ner_tags_set)])

        self.feature_dim = 2*(self.token_ner_encoder.feature_size) + \
            2*(self.pos_ner_encoder.feature_size) + \
            2*(self.chunk_ner_encoder.feature_size)
        self.ner_tags_set.difference_update([START_SYMBOL, STOP_SYMBOL])
        self.ner_tags_set = list(self.ner_tags_set)

        self.if_current_token = kwargs.get('if_current_token', True)
        self.if_prev_token = kwargs.get('if_prev_token', True)
        self.if_current_pos = kwargs.get('if_current_pos', True)
        self.if_prev_pos = kwargs.get('if_prev_pos', True)
        self.if_current_chunk = kwargs.get('if_current_chunk', True)
        self.if_prev_chunk = kwargs.get('if_prev_chunk', True)

        self.current_token_size = self.token_ner_encoder.feature_size if self.if_current_token else 0
        self.prev_token_size = self.token_ner_encoder.feature_size if self.if_prev_token else 0
        self.current_pos_size = self.pos_ner_encoder.feature_size if self.if_current_pos else 0
        self.prev_pos_size = self.pos_ner_encoder.feature_size if self.if_prev_pos else 0
        self.current_chunk_size = self.chunk_ner_encoder.feature_size if self.if_current_chunk else 0
        self.prev_chunk_size = self.chunk_ner_encoder.feature_size if self.if_prev_chunk else 0

        self.current_ner_step = 0
        self.prev_ner_step = self.current_token_size
        self.current_ner_pos_step = self.prev_ner_step + self.prev_token_size
        self.prev_ner_pos_step = self.current_ner_pos_step + self.current_pos_size
        self.current_chunk_step = self.prev_ner_pos_step + self.prev_pos_size
        self.prev_chunk_step = self.current_chunk_step + self.current_chunk_size
        import pdb; pdb.set_trace()

    def local_feature_trans(self, sentence, current_ner_tag, prev_ner_tag, k):
        current_token = STOP_SYMBOL if k == sentence.length else sentence.tokens[k]
        current_pos = STOP_SYMBOL if k == sentence.length else sentence.pos_tags[k]
        current_chunk = STOP_SYMBOL if k == sentence.length else sentence.syn_chunk[k]

        local_feature = defaultdict(float)
        if self.if_current_token: local_feature.update(self.token_ner_encoder.transform(
            sequence_ner_permutizer(current_token, current_ner_tag), step=self.current_ner_step))
        if self.if_prev_token: local_feature.update(self.token_ner_encoder.transform(
            sequence_ner_permutizer(current_token, prev_ner_tag), step=self.prev_ner_step))
        if self.if_current_pos: local_feature.update(self.pos_ner_encoder.transform(
            sequence_ner_permutizer(current_pos, current_ner_tag), step=self.current_ner_pos_step))
        if self.if_prev_pos: local_feature.update(self.pos_ner_encoder.transform(
            sequence_ner_permutizer(current_pos, prev_ner_tag), step=self.prev_ner_pos_step))
        if self.if_current_chunk: local_feature.update(self.chunk_ner_encoder.transform(
            sequence_ner_permutizer(current_chunk, current_ner_tag), step=self.current_chunk_step))
        if self.if_prev_chunk: local_feature.update(self.chunk_ner_encoder.transform(
            sequence_ner_permutizer(current_chunk, prev_ner_tag), step=self.prev_chunk_step))
        return local_feature

    def global_feature_trans(self, sentence, ner_tags):
        global_feature = defaultdict(float)
        for idx, ner_tag in enumerate(ner_tags):
            prev_ner_tag = START_SYMBOL if idx == 0 else ner_tags[idx-1]
            global_feature.update(self.local_feature_trans(sentence, ner_tag, prev_ner_tag, idx))
        return global_feature

    def score(self, w_weight_vector, sentence, current_ner_tag, prev_ner_tag, k):
        score = 0
        for idx in self.local_feature_trans(sentence, current_ner_tag, prev_ner_tag, k).keys():
            score += w_weight_vector[0,idx]
        return score

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
def sequence_ner_permutizer(sequence_input, ner):
    return "{}_vs_{}".format(sequence_input, ner)

class OneHotEncoder(object):
    def __init__(self, list_to_encode):
        self.feature_dict = dict(zip(list_to_encode, range(len(list_to_encode))))
        self.feature_size = len(list_to_encode)
    def transform(self, token, step=0):
        feature_dict = defaultdict(float)
        idx = self.feature_dict.get(token, -1)
        if not idx == -1:
            feature_dict[idx+step] += 1.
        return feature_dict
