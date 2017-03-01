import os, json, ConfigParser
import parsers.datapath_parser as dp
from Corpora import Sentence

config_path = './config.ini'
section = 'path'
data_dir = 'data_dir'

path_dict = {
    'train': dp.get_train_set_path(),
    'dev': dp.get_dev_set_path(),
    'test': dp.get_test_set_path(),
    's_train': dp.get_small_train_set_path(),
    's_dev': dp.get_small_dev_set_path(),
    's_test': dp.get_small_test_set_path()
}

def corpus_parser(*filedirs):
    corpus = []
    for filedir in list(filedirs):
        with open(filedir, 'r') as fn:
            corpus += [Sentence(token_tag_pairs.split('\n'))
                for token_tag_pairs in fn.read().split('\n\n')
                if not token_tag_pairs.split('\n')[0].startswith('-DOCSTART-')
                and not token_tag_pairs.split('\n')[0] == '']
        fn.close()
    return corpus

def get_train_corpus():
    return corpus_parser(path_dict['train'])

def get_dev_corpus():
    return corpus_parser(path_dict['dev'])

def get_test_corpus():
    return corpus_parser(path_dict['test'])

def get_small_train_corpus():
    return corpus_parser(path_dict["s_train"])

def get_small_dev_corpus():
    return corpus_parser(path_dict["s_dev"])

def get_small_test_corpus():
    return corpus_parser(path_dict["s_test"])

def get_corpus_by_tag(*tags):
    path_list = [path_dict.get(tag) for tag in list(tags)]
    if len(path_list) == 0: raise ValueError('You should only input \
        train, dev, test, s_train, s_dev or s_test corpus')
    return corpus_parser(*path_list)

if __name__ == '__main__':
    import pdb; pdb.set_trace()
