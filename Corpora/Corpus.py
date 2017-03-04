import parsers.corpus_parser as cp

class Corpus(object):
    def __init__(self, documents):
        self.documents = documents
        self.token_set = set()
        self.pos_set = set()
        self.chunk_set = set()
        self.calc_sets()
    @classmethod
    def trainCorpus(cls, is_small=False):
        documents = cp.get_train_corpus() if not is_small else cp.get_small_train_corpus()
        return cls(documents)
    @classmethod
    def devCorpus(cls, is_small=False):
        documents = cp.get_dev_corpus() if not is_small else cp.get_small_dev_corpus()
        return cls(documents)
    @classmethod
    def testCorpus(cls, is_small=False):
        documents = cp.get_test_corpus() if not is_small else cp.get_small_test_corpus()
        return cls(documents)
    @classmethod
    def combinedCorpus(cls, *tags, **kwargs):
        is_small = kwargs.pop('is_small', False)
        documents = cp.get_corpus_by_tag(*tags, is_small=is_small)
        return cls(documents)

    def __iter__(self):
        for sentence in self.documents:
            yield sentence

    def calc_sets(self):
        for sentence in self.documents:
            self.token_set.update(sentence.tokens)
            self.pos_set.update(sentence.pos_tags)
            self.chunk_set.update(sentence.syn_chunk)

    def get_sets(self):
        return self.token_set, self.pos_set, self.chunk_set

    def save(self,path):
        with open(path, 'w') as outfn:
            for sentence in self.documents:
                map(lambda k: outfn.write(
                    " ".join(k)+"\n"),
                    zip(sentence.tokens, sentence.pos_tags,
                        sentence.syn_chunk, sentence.ner,
                        list(sentence.predicted_ner_tags)))
                outfn.write('\n')
        outfn.close()
