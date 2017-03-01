class Viterbi(object):
    def __init__(self, phi_feature_trans, feature_dim, tags_vocabularies):
        self.phi_feature_trans = phi_feature_trans
        self.feature_dim = feature_dim
        self.tags_vocabularies = tags_vocabularies

    def decode(self, weight_vector, tokens):
        # TODO:
        for tag in tags:
            # TODO: find the max
