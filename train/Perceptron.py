import numpy as np
from scipy.sparse import csr_matrix

class Perceptron(object):
    def __init__(self, frm, eta=1, epoch=5, eval_every=10):
        self.frm = frm
        self.feature_transform = frm.global_feature_trans
        self.feature_dim = frm.feature_dim
        self.epoch = epoch
        self.eval_every = eval_every
        self.w_weight_vector = csr_matrix((1,self.feature_dim))
        self.w_average = csr_matrix((1,self.feature_dim))
        self.count = 0

    def fit(self,corpus):
        for loop in range(self.epoch):
            for sentence in corpus:
                predicted_ners = sentence.decode(self.frm, self.w_weight_vector)
                if not predicted_ners == sentence.ner:
                    update = self.feature_transform(sentence, sentence.ner) - \
                        self.feature_transform(sentence, predicted_ners)
                    self.w_weight_vector += update * eta
                    self.w_average += self.w_weight_vector
                    self.count += 1
        self.w_average /= self.count
        return self

    def predict(self,corpus):
        return [sentence.decode(self.frm, self.w_average) for sentence in corpus]
