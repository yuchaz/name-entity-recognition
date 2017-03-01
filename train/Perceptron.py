import numpy as np

class Perceptron(object):
    def __init__(self, viterbi, epoch, eval_every=10):
        self.viterbi = viterbi.decode
        self.phi_feature_trans = viterbi.phi_feature_trans
        self.feature_dim = viterbi.feature_dim
        # TODO: above, write the viterbi meet these specs

        self.epoch = epoch
        self.eval_every = eval_every
        # TODO: Need to think about how to randomly init w
        self.w_weight_vector = np.random.rand(self.feature_dim)
        self.w_average = np.zeros(self.feature_dim)
        self.count = 0

    def fit(X,y):
        for loop in epoch:
            for i, x_i in enumerate(X):
                y_i_predict = self.viterbi(self.w_weight_vector, x_i)
                # TODO: or regularize it, make it not always need to equal
                if not y_i_predict == y[i]:
                    update = self.phi_feature_trans(x_i, y[i]) - \
                        self.phi_feature_trans(x_i, y_i_predict)
                    self.w_weight_vector += update
                    self.w_average += self.w_weight_vector
                    self.count += 1
        self.w_average /= self.count

    def predict(X):
        return [self.viterbi(self.w_average, x_i) for x_i in X]
