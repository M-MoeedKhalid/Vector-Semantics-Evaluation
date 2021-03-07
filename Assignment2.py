from collections import defaultdict

import gensim
import pandas as pd
from gensim.models import Word2Vec, TfidfModel
import nltk
from nltk.corpus import brown

import pytrec_eval
import json

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords


def readfile(filepath, indexes=None):
    if not indexes:
        indexes = [0, 1, 2]
    file1 = open(filepath)
    count = 0
    dictionary = defaultdict(dict)

    while True:
        count += 1
        line = file1.readline()
        if not line:
            break
        file_tokens = line.strip().split()
        dictionary[str(file_tokens[indexes[0]])][str(file_tokens[indexes[1]])] = file_tokens[indexes[2]]

    file1.close()
    return dictionary


def trainword2vec(category):
    text = brown.sents(categories=category)
    w2v = Word2Vec(text, window=5, min_count=1, size=100, iter=100)
    return w2v


def normalize_dict(rank_norm, min_val, max_val):
    delta = max_val - min_val
    return [dict(d, score=(d['score'] - min_val) / delta) for d in rank_norm]


def getmapandndcg(ground_truth_model, w2v):
    # keys =
    map = 0
    ndcg = 0
    total = 0
    exists = 0
    for key in list(ground_truth_model.keys()):
        print(f"For word {key}:")
        sims = dict()
        try:
            for sim in w2v.most_similar(key, topn=10):
                sims[sim[0]] = round(sim[1] * 10, 2)

            # sims = normalize_dict(sims,1,10)
            ground_truth = ground_truth_model[key]
            change = True

            if ground_truth.__len__() < 10 and ground_truth.__len__() != 0:
                while change:
                    length = len(list(ground_truth.keys()))
                    change = False
                    for key in list(ground_truth.keys()):
                        newvalues = ground_truth_model[key]
                        if list(newvalues.keys()) not in list(ground_truth.keys()):
                            change = True
                            ground_truth.update(ground_truth_model[key])
                    if len(ground_truth.keys()) == length:
                        break
            sorted_result = dict(sorted(ground_truth.items(), key=lambda x: x[1]))
            ground_truth = {k: sorted_result[k] for k in list(sorted_result)[:10]}
            for key, value in ground_truth.items():
                ground_truth[key] = round(float(value))

            qrel = {
                'q1': ground_truth,
            }

            run = {
                'q1': sims,
            }
            evaluator = pytrec_eval.RelevanceEvaluator(
                qrel, {'map', 'ndcg'})
            print(f"ground_truth is {ground_truth}")
            print(f"prediction is {sims}")
            print(json.dumps(evaluator.evaluate(run), indent=1))
            map = map + evaluator.evaluate(run)['q1']['map']
            ndcg = ndcg + evaluator.evaluate(run)['q1']['ndcg']
            exists = exists + 1
            if evaluator.evaluate(run)['q1']['map'] > 0:
                total = total + 1
                # if evaluator.evaluate(run).q1.map !=0 or evaluator.evaluate(run).q1.ndcg !=0 :
            #     result = evaluator.evaluate(run)
        except Exception as e:
            pass
            print(repr(e))

    # print(f"Map is {map}")
    # print(f"NDCG is {ndcg}")
    return map, ndcg, total, exists


filepath = "data/WordSim-353/wordsim_similarity_goldstandard.txt"
wordsim_353 = readfile(filepath)

filepath = "data/SimLex-999/SimLex-999.txt"
simlex_999 = readfile(filepath, [0, 1, 3])

listofsents = list()
listofwords = []

print("Training on the news corpus.. ")
#################### W2V ####################
w2v = trainword2vec('news')
# ############################################
print("Training done on the news corpus")
map, ndcg, total, exists = getmapandndcg(wordsim_353, w2v)
print("News and Wordsim")
print(f'Returned values are {map} and {ndcg} on {total} values out of {exists} that exist ')

map, ndcg, total, exists = getmapandndcg(simlex_999, w2v)
print("News and Simlex")
print(f'Returned values are {map} and {ndcg} on {total} values out of {exists} that exist')

print("Training on the romance corpus.. ")
#################### W2V ####################
w2v = trainword2vec('romance')
# ############################################
print("Training done on the romance corpus")
print("Romance and Wordsim")
map, ndcg, total, exists = getmapandndcg(wordsim_353, w2v)
print(f'Returned values are {map} and {ndcg} on {total} values out of {exists} that exist')

print("Romance and Simlex")
map, ndcg, total, exists = getmapandndcg(simlex_999, w2v)
print(f'Returned values are {map} and {ndcg} on {total} values out of {exists} that exist')


# ################ Preprocessing Necessary for using tf-idf ######################


# for sent in brown.sents(categories='news'):
#     munged_sentence = ' '.join(sent).replace('``', '"').replace("''", '"').replace('`', "'")
#     sentence = TreebankWordDetokenizer().detokenize(
#         munged_sentence.split(), ).lower()
#     splits = sentence.split()
#     for split in splits:
#         listofwords.append(split)
#     listofsents.append(sentence)
# print('preprocessing done')

# ################ Simple dummy corpus ######################


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# ################TF-IDF using scikitlearn ######################

# tfIdfVectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1, 1), )
# tfIdf = tfIdfVectorizer.fit_transform(corpus)
# df = pd.DataFrame(tfIdf[0].T.toarray(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print(df[:30])
#
# df = df.to_dict()

# ################TF-IDF using NLTK ######################


# from nltk.corpus import brown
#
# text = nltk.Text(brown.words(categories="news"))
# print(text)
# print()
# print("Concordance:")
# text.concordance("news")
# print()
# print("Distributionally similar words:")
# text.similar("news")
# print()
# print("Collocations:")
# text.collocations()
# print()
# # print("Automatically generated text:")
# # text.generate()
# # print()
# print("Dispersion plot:")
# text.dispersion_plot(["news", "report", "said", "announced"])
# print()
# print("Vocabulary plot:")
# text.plot(50)
# print()
# print("Indexing:")
# print("text[3]:", text[3])
# print("text[3:5]:", text[3:5])
# print("text.vocab()['news']:", text.vocab()["news"])


# ################TF-IDF using gensim ######################

# from gensim import corpora
# from gensim.utils import simple_preprocess
#
#
# doc_tokenized = [simple_preprocess(doc) for doc in listofsents]
# dictionary = corpora.Dictionary()
# BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
# for doc in BoW_corpus:
#     print([[dictionary[id], freq] for id, freq in doc])
