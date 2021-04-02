from collections import defaultdict
import networkx as nx
from itertools import chain
import gensim
import pandas as pd
from gensim.models import Word2Vec, TfidfModel
import nltk
from nltk.corpus import brown

import pytrec_eval
import json

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords

import random
random.seed(0)
import numpy as np
np.random.seed(0)


def readfile(filepath, indexes=None, header=False):
    g = nx.Graph()
    if not indexes:
        indexes = [0, 1, 2]
    with open(filepath) as file1:
        if header: file1.readline()
        while True:
            line = file1.readline()
            if not line: break
            file_tokens = line.strip().split()
            word1, word2, score = str(file_tokens[indexes[0]]), str(file_tokens[indexes[1]]), float(file_tokens[indexes[2]])
            if score > 6:
                g.add_edges_from([(word1, word2, {'weight': score})])

    top_sim = dict()
    for word in g.nodes():
        top_sim[word] = {x:1 for x in nx.node_connected_component(g, word) if x != word}

    return top_sim

def normalize_dict(rank_norm, min_val, max_val):
    delta = max_val - min_val
    return [dict(d, score=(d['score'] - min_val) / delta) for d in rank_norm]

def getmapandndcg(ground_truth_lists, model):
    # keys =
    map = 0
    ndcg = 0
    total = 0
    exists = 0
    for key in list(ground_truth_lists.keys()):
        #print(f"For word {key}:")
        sims = dict()
        try:
            sims = model.sim_func(key)
            # sims = normalize_dict(sims,1,10)
            qrel = {'q1': ground_truth_lists[key]}
            run = {'q1': sims}
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
            #print(f"ground_truth is {ground_truth_lists[key]}")
            #print(f"prediction is {sims}")
            #print(json.dumps(evaluator.evaluate(run), indent=1))
            map = map + evaluator.evaluate(run)['q1']['map']
            ndcg = ndcg + evaluator.evaluate(run)['q1']['ndcg']
            exists = exists + 1
            if evaluator.evaluate(run)['q1']['map'] > 0:
                total = total + 1
                # if evaluator.evaluate(run).q1.map !=0 or evaluator.evaluate(run).q1.ndcg !=0 :
            #     result = evaluator.evaluate(run)
        except Exception as e:
            pass
            #print(repr(e))

    return map/exists, ndcg/exists, total, exists

def trainword2vec(category, window=5, size=100):
    text = brown.sents(categories=category)
    w2v = Word2Vec(text, window=window, min_count=1, size=size, iter=100)
    return w2v

def w2v_func(self, word):
    sims = {}
    for sim in self.most_similar(word, topn=10):
        sims[sim[0]] = sim[1]
    return sims

def w2v(gold_list):
    Word2Vec.sim_func = w2v_func
    for corpus in ['news', 'romance']:
        for gold in gold_list.keys():
            # print(f'Training on the *{corpus}* and evaluating based on *{gold}* ... ')
            # print('corpus, gold, map,ndcg,total_zero,exist_count')
            for w in range(2, 11, 1):
                for d in chain(range(10, 100, 10), range(100, 600, 100)):
                    w2v = trainword2vec(corpus, w, d)
                    map, ndcg, total, exists = getmapandndcg(gold_list[gold], w2v)
                    print(f'{corpus},{gold},{w},{d},{map},{ndcg},{total},{exists}')

import torch
def traintfvec(corpus, vec):
    tf = vec.fit_transform([' '.join(s) for s in brown.sents(categories=corpus)])
    sim = 1 - metrics.pairwise_distances(tf.transpose(), metric='cosine', n_jobs=-1)
    vec.scores, vec.idxes = torch.tensor(sim).topk(10, dim=1)
    vec.word2idx = vec.vocabulary_
    vec.idx2word = {v: k for k, v in vec.vocabulary_.items()}

def tfvec_func(self, word):
    row = self.word2idx[word]
    sims = {self.idx2word[x]: self.scores[row, i].item() for i, x in enumerate(self.idxes[row, :].tolist()) if word !=self.idx2word[x]}
    return sims

def tf_idf(gold_list):
    for corpus in ['news', 'romance']:
        CountVectorizer.sim_func = tfvec_func
        TfidfVectorizer.sim_func = tfvec_func
        for vec in [CountVectorizer(ngram_range=(1, 1)), TfidfVectorizer(use_idf=True, ngram_range=(1, 1), norm='l2')]:
            traintfvec(corpus, vec)#everything needed is attached to the obj instance of vec :D
            for gold in gold_list.keys():
                map, ndcg, total, exists = getmapandndcg(gold_list[gold], vec)
                print(f'{corpus},{gold},{vec},{map},{ndcg},{total},{exists}')


if __name__ == "__main__":
    gold_list = dict()

    filepath = "data/WordSim-353/wordsim_similarity_goldstandard.txt"
    gold_list['wordsim_353'] = readfile(filepath)

    filepath = "data/SimLex-999/SimLex-999.txt"
    gold_list['simlex_999'] = readfile(filepath, [0, 1, 3], header=True)

    #w2v(gold_list)
    #tf_idf(gold_list)
