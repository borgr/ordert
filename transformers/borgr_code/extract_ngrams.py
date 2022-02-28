from collections import Counter

from nltk.util import ngrams
path = r"/cs/labs/daphna/guy.hacohen/borgr/ordert/data/train.en"
path = "README"
maxgram = 6
lm = [Counter() for i in range(1, maxgram)]

with open(path) as fl:
    for line in fl:
        for n in range(1, maxgram):
            ngram = ngrams(line.split(), n)
