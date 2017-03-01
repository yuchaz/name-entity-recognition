class Viterbi(object):
    def __init__(self, fmm):
        self.fmm = fmm
        self.predicted_ner_tags = deque([])
        self.hidden_cells = [{} for i in range(self.doc_length+2)]
        self.hidden_cells[0] = {START_SYMBOL:Cell(1,'')}

    def decode(sentence, end_index=0):
        if end_index <= len(sentence):
            tag_choices = fmm.possible_tag_choices \
                if end_index != len(sentence) else [STOP_SYMBOL]

            for target_tag in tag_choices:
                self.calc_pi_score(target_tag, sentence, end_index)
            self.decode(sentence, end_index+1)
        else:
            self.back_propagation(end_index, len(sentence))

    def calc_pi_score(self, current_ner_tag, sentence, end_index):
        current_idx = end_index+1
        current_token = None if end_index == len(sentence) else sentence[end_index]
        if end_index == 0:
            # max_pi_score = fmm.e_score(TokenPosTagPair(current_token,current_ner_tag)) + \
            #                fmm.q_score((START_SYMBOL,current_ner_tag))
            predicted_prev_tag = START_SYMBOL
        elif end_index == len(sentence):
            predicted_prev_tag, max_pi_score = \
                max(((bio_tag, fmm.q_score((pos_tag,current_ner_tag)) + \
                               self.hidden_cells[current_idx-1][ner_tag].score)
                               for ner_tag in self.fmm.possible_tag_choices),
                               key=lambda p:p[1])
        else:
            predicted_prev_tag, max_pi_score = \
                # max(((pos_tag, fmm.e_score(TokenPosTagPair(current_token,current_ner_tag)) + \
                #                fmm.q_score((pos_tag,current_ner_tag)) + \
                #                self.hidden_cells[current_idx-1][pos_tag].score)
                #                for pos_tag in fmm.possible_pos_choices),
                #                key=lambda p:p[1])
        self.store_cells(current_idx, current_ner_tag, max_pi_score, predicted_prev_tag)

    def store_cells(self,idx,ner_tag,score,argmax_ner_tag):
        self.hidden_cells[idx].update({ner_tag: Cell(score, argmax_ner_tag)})

    def back_propagation(self, end_index, sentence_len):
        if end_index > sentence_len:
            back_propagation(end_index-1)

        elif end_index > 0:
            current_ner_tag = STOP_SYMBOL if end_index == sentence_len
                                          else self.predicted_ner_tags[0]
            score, back_ner_tag = self.hidden_cells[end_index+1][current_ner_tag].get_tuples()
            self.predicted_ner_tags.appendleft(back_ner_tag)
            self.back_propagation(end_index-1, sentence_len)
        else:
            return

class TokenPosTagPair(object):
    def __init__(self, token, pos_tag):
        self.token = token
        self.pos_tag = pos_tag
    def get_tuples(self):
        return self.token, self.pos_tag
    def replaced_with(self,to_unk_dict):
        unk_type = to_unk_dict.get(self.token)
        if unk_type != None:
            self.token = unk_type

class Cell(object):
    def __init__(self, score, back_ner_tag):
        self.score = score
        self.back_ner_tag = back_ner_tag
    def __str__(self):
        return str(self.get_tuples())
    def get_tuples(self):
        return self.score, self.back_ner_tag
