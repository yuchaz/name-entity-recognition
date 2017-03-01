from Corpora import Corpus

def main():
    corpus = Corpus.trainCorpus()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    try:
        import time
        start = time.time()
        main()
        print time.time()-start
    except:
        import sys, pdb, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
