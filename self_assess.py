#!/usr/bin/python3

import nltk
from nltk.collocations import BigramAssocMeasures,BigramCollocationFinder
from nltk.corpus import brown, stopwords
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


P = ['voyage', 'monk', 'stove', 'grin', 'furnace', 'mound', 'cushion', 'pillow', 'graveyard', 'cord',
     'lad', 'signature', 'crane', 'cock', 'midday', 'implement', 'magician', 'gem', 'rooster',
     'autograph', 'wizard', 'woodland', 'oracle', 'sage', 'tumbler', 'asylum', 'madhouse', 'jewel']
S = {
        ('cord', 'smile'): 0.02,
        ('rooster', 'voyage'): 0.04,
        ('noon', 'string'): 0.04,
        ('fruit', 'furnace'): 0.05,
        ('autograph', 'shore'): 0.06,
        ('automobile', 'wizard'): 0.11,
        ('mound', 'stove'): 0.14,
        ('grin', 'implement'): 0.18,
        ('asylum', 'fruit'): 0.19,
        ('asylum', 'monk'):0.39,
        ('graveyard', 'madhouse'): 0.42,
        ('glass', 'magician'): 0.44,
        ('boy', 'rooster'): 0.44,
        ('cushion', 'jewel'): 0.45,
        ('monk', 'slave'): 0.57,
        ('asylum', 'cemetary'): 0.79,
        ('coast', 'forest'): 0.85,
        ('grin', 'lad'): 0.88,
        ('shore', 'voyage'): 1.22,
        ('bird', 'woodland'): 1.24,
        ('coast', 'hill'): 1.26,
        ('furnace', 'implement'): 1.37,
        ('crane', 'rooster'): 1.41,
        ('hill', 'woodland'): 1.48,
        ('car', 'journey'): 1.55,
        ('cemetary', 'mound'): 1.69,
        ('glass', 'jewel'): 1.78,
        ('magician', 'oracle'): 1.82,
        ('crane', 'implement'): 2.37,
        ('brother', 'lad'): 2.41,
        ('sage', 'wizard'): 2.46,
        ('oracle', 'sage'): 2.61,
        ('bird', 'crane'): 2.63,
        ('bird', 'cock'): 2.63,
        ('food', 'fruit'): 2.69,
        ('brother', 'monk'): 2.74,
        ('asylum', 'madhouse'): 3.04,
        ('furnace', 'stove'): 3.11,
        ('magician', 'wizard'): 3.21,
        ('hill', 'mound'): 3.29,
        ('cord', 'string'): 3.41,
        ('glass', 'tumbler'): 3.45,
        ('grin', 'smile'): 3.46,
        ('serf', 'slave'): 3.46,
        ('journey', 'voyage'): 3.58,
        ('autograph', 'signature'): 3.59,
        ('coast', 'shore'): 3.60,
        ('forest', 'woodland'): 3.65,
        ('implement', 'tool'): 3.66,
        ('cock', 'rooster'): 3.68,
        ('boy', 'lad'): 3.82,
        ('cushion', 'pillow'): 3.84,
        ('cemetary', 'graveyard'): 3.88,
        ('automobile', 'car'): 3.92,
        ('midday', 'noon'): 3.94,
        ('gem', 'jewel'): 3.94
    }


def extract_words():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    nltk.download("brown")
    additions = ['cord', 'smile', 'rooster', 'voyage', 'noon', 'string', 'furnace',
        'autograph', 'shore', 'automobile', 'wizard', 'mound', 'stove', 'grin', 'implement',
        'asylum', 'fruit', 'monk', 'graveyard', 'madhouse', 'glass', 'magician', 'boy',
        'cushion', 'jewel', 'slave', 'cemetary', 'coast', 'forest', 'lad',
        'woodland', 'oracle', 'sage', 'food', 'bird', 'hill', 'crane', 'car', 'journey',
        'brother', 'cock', 'tumbler', 'serf', 'signature', 'tool', 'pillow', 'midday', 'gem']

    freq = nltk.FreqDist(word.lower() for word in brown.words() if word.isalpha() and word not in stop_words)
    W = freq.most_common(5000)
    words = [entry[0] for entry in W]

    five_most = W[:5]
    five_least = W[-5:]
    least_count = W[-1][1]

    new_entries = []
    for entry in additions:
         if freq[entry] > 0 and freq[entry] < least_count and entry not in words:
             new_entries.append((entry, freq[entry]))
             words.append(entry)
    new_entries.sort(key=lambda element: element[1], reverse=True)
    W.extend(new_entries)

    list_size = len(W)
    print(new_entries)
    print("Five most common words")
    print(five_most)
    print("Five least common words")
    print(five_least)
    print("|W| = {}".format(list_size))

    return W, words, freq


def build_word_context_model(words):
    finder = nltk.BigramCollocationFinder.from_words(word.lower() for word in brown.words())
    bigram_filter = lambda w1,w2: not w1.isalpha() or not w2.isalpha() or w1 not in words or w2 not in words
    finder.apply_ngram_filter(bigram_filter)

    all_rows = []
    all_ordered = []
    for word_1 in words:
        row = []
        ordered = []
        for word_2 in words:
            row.append(finder.ngram_fd[(word_1, word_2)] + finder.ngram_fd[(word_2, word_1)])
            ordered.append(finder.ngram_fd[(word_1, word_2)])
        all_rows.append(row)
        all_ordered.append(ordered)

    df = pd.DataFrame(all_rows, columns=words, index=words)
    df_ordered = pd.DataFrame(all_ordered, columns=words, index=words)
    return df,df_ordered,finder


def compute_ppmi(finder, words):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    scores = {entry[0]:entry[1] for entry in finder.score_ngrams(bigram_measures.pmi)}

    all_rows = []
    for word_1 in words:
        row = []
        for word_2 in words:
            if (word_1, word_2) not in scores or scores[(word_1, word_2)] < 0:
                row.append(0)
            else:
                row.append(scores[(word_1, word_2)])
        all_rows.append(row)

    df = pd.DataFrame(all_rows, columns=words, index=words)
    return df


def apply_pca(df, num_components, words):
    pca = PCA(n_components = num_components)
    model = pca.fit_transform(df)
    return pd.DataFrame(data=model, index=words)


def evaluate_cosine(model_df):
    ret = {}
    for (w1,w2) in S.keys():
        if w1 in P and w2 in P:
            v1 = model_df.loc[w1]
            v2 = model_df.loc[w2]
            in1 = v1.to_numpy().reshape(1,v1.shape[0])
            in2 = v2.to_numpy().reshape(1,v2.shape[0])
            res = cosine_similarity(in1, in2)
            ret[(w1,w2)] = res[0][0]
    return ret


def calc_pearson(results):
    model = []
    truth = []
    for k,v in results.items():
        if k in S:
            model.append(v)
            truth.append(S[k])

    return pearsonr(model, truth)


if __name__ == "__main__":
    W,words,freq = extract_words()
    M1,M1_ordered,finder = build_word_context_model(words)
    M1_plus = compute_ppmi(finder, words)
    M2_10 = apply_pca(M1_plus, 10, words)
    M2_100 = apply_pca(M1_plus, 100, words)
    M2_300 = apply_pca(M1_plus, 300, words)

    SM1 = evaluate_cosine(M1)
    SM1_ordered = evaluate_cosine(M1_ordered)
    SM1_plus = evaluate_cosine(M1_plus)
    SM2_10 = evaluate_cosine(M2_10)
    SM2_100 = evaluate_cosine(M2_100)
    SM2_300 = evaluate_cosine(M2_300)

    print("bigram count correlation {}".format(calc_pearson(SM1)[0]))
    print("bigram ordered count correlation {}".format(calc_pearson(SM1_ordered)[0]))
    print("ppmi correlation {}".format(calc_pearson(SM1_plus)[0]))
    print("pca_10 correlation {}".format(calc_pearson(SM2_10)[0]))
    print("pca_100 correlation {}".format(calc_pearson(SM2_100)[0]))
    print("pca_300 correlation {}".format(calc_pearson(SM2_300)[0]))

