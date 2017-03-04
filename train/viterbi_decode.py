import numpy as np

def viterbi_decode(sentence,initialProb,transProb):
    n = initialProb.shape[0]
    trellis = np.zeros((n, len(sentence)))
    backpt = np.ones((n, len(sentence)), 'int32') * -1
    trellis[:,0] = initialProb
    for t in xrange(1,len(sentence)):
        trellis[:, t] = (trellis[:, t-1, None] + transProb).max(0)
        backpt[:, t] = (trellis[:, t-1, None] + transProb).argmax(0)
    tokens = [trellis[:, -1].argmax()]
    import pdb; pdb.set_trace()
    for i in xrange(len(sentence)-1, 0, -1):
        tokens.append(backpt[tokens[-1], i])
    return tokens[::-1]
