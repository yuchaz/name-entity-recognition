import parsers.corpus_parser as cp

class Corpus(object):
    def __init__(self, documents):
        self.documents = documents
    @classmethod
    def trainCorpus(cls):
        documents = cp.get_train_corpus()
        return cls(documents)
    @classmethod
    def devCorpus(cls):
        documents = cp.get_dev_corpus()
        return cls(documents)
    @classmethod
    def testCorpus(cls):
        documents = cp.get_test_corpus()
        return cls(documents)
    @classmethod
    def sTrainCorpus(cls):
        documents = cp.get_small_train_corpus()
        return cls(documents)
    @classmethod
    def sDevCorpus(cls):
        documents = cp.get_small_dev_set_path()
        return cls(documents)
    @classmethod
    def sTestCorpus(cls):
        documents = cp.get_small_test_set_path()
        return cls(documents)
    @classmethod
    def combinedCorpus(cls, *tags):
        documents = cp.get_corpus_by_tag(*tags)
        return cls(documents)

    def __iter__(self):
        for doc in self.documents:
            yield doc
