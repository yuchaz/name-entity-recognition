from collections import deque
START_SYMBOL = '<START>'
STOP_SYMBOL = '<STOP>'
import time

class Sentence(object):
    def __init__(self, tokens_tags_pairs):
        self.tokens, self.pos_tags, self.syn_chunk, self.ner = \
            zip(*[token_tag.split(' ') for token_tag in tokens_tags_pairs])
        self.length = len(self.tokens)
        self.all_tokens = zip(self.tokens, self.pos_tags, self.syn_chunk)
        self.hidden_cells = [{} for i in range(self.length+2)]
        self.hidden_cells[0] = {START_SYMBOL:Cell(1,'')}
        self.predicted_ner_tags = deque([])

    def __len__(self):
        return self.length
    def __iter__(self):
        for idx in range(self.length):
            return self.tokens[idx], self.pos_tags[idx], self.syn_chunk[idx]

    def get_all_tokens(self, index):
        return self.all_tokens[index]

    def init_before_decode(self):
        self.predicted_ner_tags = deque([])
        self.hidden_cells = [{} for i in range(self.length+2)]
        self.hidden_cells[0] = {START_SYMBOL:Cell(1,'')}

    def decode(self, frm, w_weight_vector):
        self.init_before_decode()
        self.viterbi_decode(frm, w_weight_vector)
        return list(self.predicted_ner_tags)

    def viterbi_decode(self, frm, w_weight_vector, end_index=0):
        if end_index <= self.length:
            tag_choices = frm.ner_tags_set \
                if end_index != self.length else [STOP_SYMBOL]

            for target_tag in tag_choices:
                self.calc_pi_score(target_tag, frm, end_index, w_weight_vector)
            self.viterbi_decode(frm, w_weight_vector, end_index+1)
        else:
            self.back_propagation(end_index)

    def calc_pi_score(self, current_ner_tag, frm, end_index, w_weight_vector):
        current_idx = end_index+1
        if end_index == 0:
            max_pi_score = frm.score(w_weight_vector, self, current_ner_tag, START_SYMBOL, end_index)
            predicted_prev_tag = START_SYMBOL
        else:
            predicted_prev_tag, max_pi_score = \
                max(((ner_tag, frm.score(w_weight_vector, self, current_ner_tag, ner_tag, end_index) + \
                               self.hidden_cells[current_idx-1][ner_tag].score)
                               for ner_tag in frm.ner_tags_set),
                               key=lambda p:p[1])
        self.store_cells(current_idx, current_ner_tag, max_pi_score, predicted_prev_tag)

    def store_cells(self,idx,ner_tag,score,argmax_ner_tag):
        self.hidden_cells[idx].update({ner_tag: Cell(score, argmax_ner_tag)})

    def back_propagation(self, end_index):
        if end_index > self.length:
            self.back_propagation(end_index-1)

        elif end_index > 0:
            current_ner_tag = STOP_SYMBOL if end_index == self.length \
                                          else self.predicted_ner_tags[0]
            score, back_ner_tag = self.hidden_cells[end_index+1][current_ner_tag].get_tuples()
            self.predicted_ner_tags.appendleft(back_ner_tag)
            self.back_propagation(end_index-1)
        else:
            return


class Cell(object):
    def __init__(self, score, back_ner_tag):
        self.score = score
        self.back_ner_tag = back_ner_tag
    def __str__(self):
        return str(self.get_tuples())
    def get_tuples(self):
        return self.score, self.back_ner_tag
