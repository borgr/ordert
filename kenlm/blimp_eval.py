import json
import os
import re

import kenlm
import spacy

en_nlp = spacy.load("en")
import numpy as np

BLIMP = "/cs/snapless/oabend/borgr/ordert/blimp/data"


def unigram_sentence_prob(log_probs, words, default_log_prob=None):
    log_prob = 0
    if default_log_prob is None:
        default_log_prob = min(log_probs.values()) - 0.000001
    for word in words:
        log_prob += log_probs.get(word, default_log_prob)
    return log_prob


def score_unigram(path):
    print(f"calculating for unigrams in {path}")

    model = {}
    with open(path) as fl:
        log_probs = False
        for line in fl:
            if line.startswith(r"\1-grams"):
                log_probs = True
            elif line.startswith(r"\end"):
                log_probs = False
            elif line.strip() and log_probs:
                # try:
                line = line.strip().split()
                prob = line[0]
                word = " ".join(line[1:])
                # except ValueError as e:
                #     print(line.strip().split())
                model[word] = float(prob)
    assert unigram_sentence_prob(model, "the the the the") > unigram_sentence_prob(model, "breakfast is best meal")
    assert unigram_sentence_prob(model, "the the the the") > unigram_sentence_prob(model, "breakfast sda waso dsao")
    arr = []
    for root, dirs, files in os.walk(BLIMP):
        for name in files:
            if "jsonl" not in name:
                continue
            print(name)
            f = open(os.path.join(root, name))
            lines = f.readlines()
            correct = 0
            total = 0
            for l in lines:
                l = json.loads(l)
                bads = [str(x) for x in en_nlp.tokenizer(
                    re.sub(r"\n+", "\n", l['sentence_bad']).replace("\n", " "))]
                goods = [str(x) for x in en_nlp.tokenizer(re.sub(r"\n+", "\n", l['sentence_good']).replace("\n", " "))]
                # print((goods, bads, (len(goods), len(bads))))
                # assert len(goods) == len(bads), (goods, bads, (len(goods), len(bads)))
                bp = unigram_sentence_prob(model, bads)
                gp = unigram_sentence_prob(model, goods)
                if gp/len(goods) > bp/len(bads):  # compare average token probability as sometimes tokens are split
                    correct += 1
                total += 1
            print(str(correct) + " / " + str(total))
            arr.append(correct)
    print(f"Final unigram results: {np.average(arr)}")


def main(model_name):
    model = kenlm.Model(model_name)
    print(f"calculating for {model_name}")
    arr = []
    for root, dirs, files in os.walk(BLIMP):
        for name in files:
            if "jsonl" not in name:
                continue
            print(name)
            f = open(os.path.join(root, name))
            lines = f.readlines()
            correct = 0
            total = 0
            for l in lines:
                l = json.loads(l)
                bads = " ".join([str(x) for x in en_nlp.tokenizer(
                    re.sub(r"\n+", "\n", l['sentence_bad']).replace("\n", " "))])  # ;print(bads)
                goods = " ".join(
                    [str(x) for x in en_nlp.tokenizer(re.sub(r"\n+", "\n", l['sentence_good']).replace("\n", " "))])
                bp = model.score(bads, bos=False, eos=False)
                gp = model.score(goods, bos=False, eos=False)
                if gp > bp:
                    correct += 1
                total += 1
            print(str(correct) + " / " + str(total))
            arr.append(correct)
    print(f"Final {model_name} results: {np.average(arr)}")


if __name__ == '__main__':
    dir = "/cs/labs/daphna/guy.hacohen/borgr/ordert/"
    models = ['egw2.arpa', 'egw3.arpa', 'egw4.arpa', 'egw5.arpa']
    models = [os.path.join(dir, model) for model in models]
    score_unigram(os.path.join(dir, "egw1.arpa"))

    # for model in models:
    #     main(model)
def score(model, sentence):
    good = " ".join(
        [str(x) for x in en_nlp.tokenizer(sentence)])
    bp = model.score(good, bos=False, eos=False)
    return bp