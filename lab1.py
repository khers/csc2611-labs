#!/usr/bin/python3

import nltk
from nltk.collocations import BigramAssocMeasures,BigramCollocationFinder
from nltk.corpus import brown, stopwords
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from multiprocessing import Pool

from self_assess import extract_words,build_word_context_model,compute_ppmi
from self_assess import apply_pca,evaluate_cosine,calc_pearson

tests = []
global_model = None

def evaluate_analogies(index):
    model,test = tests[index]
    return model.evaluate_word_analogies(test)


def run_tests(inputs):
    ret = dict()
    correct = 0
    total = 0
    wrong = []
    model = global_model
    for i in inputs:
        total += 1
        v = np.add(model[i[2]], np.subtract(model[i[1]], model[i[0]]))
        results = model.similar_by_vector(v, topn=4)
        for entry in results:
            if entry[0] in i[:3]:
                continue
            if entry[0] == i[3]:
                correct += 1
            else:
                wrong.append((i[0], i[1], i[2], i[3], entry[0]))
            break
    ret['correct'] = correct
    ret['total'] = total
    ret['wrong'] = wrong
    return ret


def local_analogies(index):
    inputs = []
    global global_model
    global_model,test = tests[index]
    with open(test, 'r') as f:
        for line in f:
            words = line.split()
            if len(words) is not 4:
                continue
            inputs.append(words)
    results = [run_tests(inputs),]

    total = 0
    correct = 0
    wrong = []
    for entry in results:
        total += entry['total']
        correct += entry['correct']
        wrong.extend(entry['wrong'])

    print("Got {} of {} which is accuracy {}".format(correct, total, (correct * 1.0) / total))
    print("Incorrect entries: {}".format(wrong))


if __name__ == "__main__":
    _,words,freq = extract_words(5000)
    _,_,finder = build_word_context_model(words)
    M1_plus = compute_ppmi(finder, words)
    M2_300 = apply_pca(M1_plus, 300, words)
    W2V = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    SM2_300 = evaluate_cosine(M2_300)
    print("pca_300 correlation {}".format(calc_pearson(SM2_300)[0]))
    SW2V = evaluate_cosine(W2V, False)
    print("word2vec correlation {}".format(calc_pearson(SW2V)[0]))

    SM_keyed = WordEmbeddingsKeyedVectors(300)
    SM_keyed.add(words, M2_300.to_numpy())

    tests = [(W2V,'./word-test.v1.txt'),
            (W2V,'./filtered-test.txt'),
            (SM_keyed,'./word-test.v1.txt'),
            (SM_keyed,'./filtered-test.txt')]

    inputs = [0, 1, 2, 3]
    with Pool(8) as p:
        results = p.map(evaluate_analogies, inputs)

    print("loaded model has accuracy {} on full analogies".format(results[0][0]))
    print("computed model has accuracy {} on full analogies".format(results[2][0]))

    print("loaded model has accuracy {} on filtered analogies".format(results[1][0]))
    print("computed model has accuracy {} on filtered analogies".format(results[3][0]))

