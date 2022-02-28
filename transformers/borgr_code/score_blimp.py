import json
import os
import re
import numpy as np

import spacy
from benepar.spacy_plugin import BeneparComponent

en_nlp = spacy.load("en")
en_nlp.add_pipe(BeneparComponent('benepar_en'))
doc = en_nlp('The time for action is now. Its never too late to do something.')

BLIMP = "/cs/snapless/oabend/borgr/ordert/blimp/data"

def syn_depth(sent):
    doc = en_nlp(sent)
    sent = list(doc.sents)[0]
    max_depth = 0
    cur = 0
    for char in sent._.parse_string:
        if char == "(":
            cur += 1
            max_depth = max(max_depth, cur)
        elif char == ")":
            cur -= 1
    return max_depth
    # print(sent._.parse_string)
    # # (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
    # print(sent._.labels)
    # # ('S',)
    # print(list(sent._.children)[0])
    # # The time for action


def score_challenges(metric_name, metric, aggr=max):
    print(f"calculating for {metric_name}")
    arr = []
    for root, dirs, files in os.walk(BLIMP):
        for name in files:
            if "jsonl" not in name:
                continue
            print(name)
            f = open(os.path.join(root, name))
            lines = f.readlines()
            sum = 0
            total = 0
            for l in lines:
                l = json.loads(l)
                bad = " ".join([str(x) for x in en_nlp.tokenizer(
                    re.sub(r"\n+", "\n", l['sentence_bad']).replace("\n", " "))])  # ;print(bads)
                good = " ".join(
                    [str(x) for x in en_nlp.tokenizer(re.sub(r"\n+", "\n", l['sentence_good']).replace("\n", " "))])
                # bp = [metric(bad) for bad in bads]
                # gp = [metric(good) for good in goods]
                bp = metric(bad)
                gp = metric(good)
                sum += aggr(bp, gp)
                total += 1
            print(str(sum) + " / " + str(total))
            arr.append(sum)
    print(f"Final {metric_name} results: {np.average(arr)}")


if __name__ == '__main__':
    dir = "/cs/labs/daphna/guy.hacohen/borgr/ordert/"
    metrics = [("sen_dept", syn_depth), ("len", len)]
    for metric_name, metric in metrics:
        score_challenges(metric_name, metric)
