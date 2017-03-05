import parsers.corpus_parser as cp
import numpy as np
import matplotlib.pyplot as plt

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

    def plot_confusion_matrix(self,frm):
        self.get_all_confusion_matrix(frm)
        self.save_confusion_matrix(frm.ner_tags_set)

    def get_all_confusion_matrix(self,frm):
        vocab_size = len(frm.ner_tags_set)
        all_confusion_matrix = np.zeros((vocab_size,vocab_size))
        for sentence in self.documents:
            confusion_matrix = sentence.confusion_matrix(frm)
            all_confusion_matrix = np.add(all_confusion_matrix, confusion_matrix)
        self.confusion_matrix = all_confusion_matrix
        return self.confusion_matrix

    def save_confusion_matrix(self, classes):
        norm_conf = []
        for i in self.confusion_matrix:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                if a == 0:
                    tmp_arr.append(0)
                else:
                    tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                        interpolation='nearest')

        width, height = self.confusion_matrix.shape

        cb = fig.colorbar(res)
        plt.xticks(range(width), classes)
        plt.yticks(range(height), classes)
        plt.savefig('confusion_matrix.png', format='png')
