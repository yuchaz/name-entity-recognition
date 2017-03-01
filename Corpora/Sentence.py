class Sentence(object):
    def __init__(self, tokens_tags_pairs):
        self.tokens, self.pos_tags, self.syn_chunk, self.ner = \
            zip(*[token_tag.split(' ') for token_tag in tokens_tags_pairs])

    def __len__(self):
        return len(self.tokens)
