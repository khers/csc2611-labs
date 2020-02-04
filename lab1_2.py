#!/usr/bin/python3

import numpy as np
import pickle
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from scipy.stats import pearsonr
import statistics
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


words = []


def read_and_organize_models(path):
    global words
    stored = {}
    with open(path, 'rb') as f:
        stored = pickle.load(f)

    decade_index = 0
    models_by_decade = {}
    for decade in stored['d']:
        all_rows = []
        word_index = 0
        for word in stored['w']:
            all_rows.append(stored['E'][word_index][decade_index])
            word_index += 1
        models_by_decade[decade] = pd.DataFrame(all_rows, index=stored['w'])
        decade_index += 1
    words = stored['w']
    return models_by_decade


def compute_rotated_similarity(early, later):
    R,_ = orthogonal_procrustes(early, later, check_finite=False)
    rotated = early @ R
    sims = cosine_similarity(rotated.to_numpy(), later.to_numpy())
    ret = list()
    for i in range(0, len(sims)):
        ret.append(sims[i][i])
    return ret


def get_similar_set(word, model):
    similar = set()
    values = model.most_similar(word, topn=20)
    for v in values:
        similar.add(v[0])
    return similar


def compute_similar_nn(early, later):
    M_early = WordEmbeddingsKeyedVectors(300)
    M_early.add(words, early.to_numpy())
    M_later = WordEmbeddingsKeyedVectors(300)
    M_later.add(words, later.to_numpy())

    scores = list()
    for word in words:
        early_similar = get_similar_set(word, M_early)
        later_similar = get_similar_set(word, M_later)
        count = len(early_similar.intersection(later_similar))
        scores.append(count)
    return scores


def compute_scaled_average_sim(early, later):
    M_early = WordEmbeddingsKeyedVectors(300)
    M_early.add(words, early.to_numpy())
    M_later = WordEmbeddingsKeyedVectors(300)
    M_later.add(words, later.to_numpy())

    scores = list()
    for word in words:
        score = 0
        early_values = M_early.most_similar(word, topn=20)
        later_values = M_later.most_similar(word, topn=20)
        early_dict = {v[0]:v[1] for v in early_values}
        later_dict = {v[0]:v[1] for v in later_values}
        overlap = set([w for w in early_dict.keys() if w in later_dict])
        early_avg = 0
        later_avg = 0
        for entry in overlap:
            early_avg += early_dict[entry]
            later_avg += later_dict[entry]
        early_avg = early_avg / len(overlap) if len(overlap) else 0
        later_avg = later_avg / len(overlap) if len(overlap) else 0
        scores.append(len(overlap) + (1 - abs(later_avg - early_avg)))
    return scores


def score_instance(word, early, later):
    R,_ = orthogonal_procrustes(early, later, check_finite=False)
    rotated = early @ R
    return cosine_similarity(rotated.loc[word].to_numpy().reshape(1,300),
                            later.loc[word].to_numpy().reshape(1,300))[0][0]


def score_decades(models, word):
    decade_pairs = [(1900,1910), (1910, 1920), (1920,1930), (1930,1940), (1940,1950), (1950,1960),
                 (1960,1970), (1970,1980), (1980,1990)]
    scores = list()
    for pair in decade_pairs:
        scores.append(score_instance(word, models[pair[0]], models[pair[1]]))
    return scores


def detect_change(scores, word):
    change = scores[0]
    stddev = statistics.stdev(scores)
    for i in range(1, len(scores)):
        if abs(scores[i] - change) > stddev:
            change = scores[i]
            print("change detected at {} for {}".format((1900 + i * 10), word))


if __name__ == "__main__":
    models_by_decade = read_and_organize_models('./data.pkl')
    cosine_sim_scores = compute_rotated_similarity(models_by_decade[1900], models_by_decade[1990])
    nn_count_scores = compute_similar_nn(models_by_decade[1900], models_by_decade[1990])
    scaled_avg_scores = compute_scaled_average_sim(models_by_decade[1900], models_by_decade[1990])

    print("Pearson correlation cosine to NN overlap count {}".format(pearsonr(cosine_sim_scores, nn_count_scores)[0]))
    print("Pearson correlation cosine to scaled overlap {}".format(pearsonr(cosine_sim_scores, scaled_avg_scores)[0]))
    print("Pearson correlation NN overlap count to scaled overlap {}".format(pearsonr(nn_count_scores, scaled_avg_scores)[0]))

    cos = list(zip(words, cosine_sim_scores))
    nn = list(zip(words, nn_count_scores))
    scale = list(zip(words, scaled_avg_scores))

    cos.sort(key=lambda v: v[1], reverse=True)
    nn.sort(key=lambda v: v[1], reverse=True)
    scale.sort(key=lambda v: v[1], reverse=True)
    print("rotated cosine twenty least changed {}".format([w[0] for w in cos[:20]]))
    print("rotated cosine twenty most changed {}\n".format([w[0] for w in cos[-20:]]))

    print("nearest neighbor set twenty least changed {}".format([w[0] for w in nn[:20]]))
    print("nearest neighbor set most changed {}\n".format([w[0] for w in nn[-20:]]))

    print("nearest neighbor set + avg distance twenty least changed {}".format([w[0] for w in scale[:20]]))
    print("nearest neighbor set + avg distance most changed {}".format([w[0] for w in scale[-20:]]))

    c_d = dict(cos)
    n_d = dict(nn)
    s_d = dict(scale)

    raw_truth = ["america 1", "finger 6", "file 8", "towns 2", "purchase 3", "particles 0", "earth 4",
            "prisoner 1", "goal 2", "artist 2", "sciences 4", "bill 5", "corn 2", "anyone 0",
            "foundation 3", "opportunity 4", "tongue 5", "haven 1", "meal 5", "regret 0", "appointment 1",
            "zealand 0", "efforts 1", "code 10", "attack 7", "urine 0", "test 7", "surprise 4", "document 1",
            "type 6", "windows 9", "habit 4", "breast 2", "harvard 0", "assistance 2", "end 3", "use 6",
            "pacific 2", "respects 3", "procedure 2", "printing 6", "character 4", "smoke 5", "remainder 1",
            "world 4", "manners 3", "clergy 0", "quarters 5", "faith 4", "speaker 6"]
    truth_words = list()
    truth_scores = list()
    for item in raw_truth:
        word,score = item.split()
        truth_words.append(word)
        truth_scores.append(int(score))

    c_scores = list()
    n_scores = list()
    s_scores = list()

    for word in truth_words:
        c_scores.append(c_d[word])
        n_scores.append(n_d[word])
        s_scores.append(s_d[word])

    print("Pearson with cos {}".format(pearsonr(c_scores, truth_scores)[0]))
    print("Pearson with NN {}".format(pearsonr(n_scores, truth_scores)[0]))
    print("Pearson with NN + avg {}".format(pearsonr(s_scores, truth_scores)[0]))

    ml_scores = score_decades(models_by_decade, 'ml')
    mcgraw_scores = score_decades(models_by_decade, 'mcgraw')
    skills_scores = score_decades(models_by_decade, 'skills')

    detect_change(ml_scores, 'ml')
    detect_change(mcgraw_scores, 'mcgraw')
    detect_change(skills_scores, 'skills')

    x_axis = [1910,1920,1930,1940,1950,1960,1970,1980,1990]

    plt.ylabel('Cosine Similarity')
    plt.xlabel('Decade')
    plt.plot(x_axis, ml_scores, color='b')
    plt.axvline(x=1930, color='r')
    r_patch = mlines.Line2D([],[],color='r', label='Change Point')
    b_patch = mlines.Line2D([],[],color='b', label='Cosine Similarity')
    plt.legend(handles=[r_patch,b_patch])
    plt.savefig("ml.pdf")
    plt.clf()

    plt.ylabel('Cosine Similarity')
    plt.xlabel('Decade')
    plt.plot(x_axis, mcgraw_scores, color='b')
    plt.axvline(x=1920, color='r')
    r_patch = mlines.Line2D([],[],color='r', label='Change Point')
    b_patch = mlines.Line2D([],[],color='b', label='Cosine Similarity')
    plt.legend(handles=[r_patch,b_patch])
    plt.savefig("mcgraw.pdf")
    plt.clf()

    plt.ylabel('Cosine Similarity')
    plt.xlabel('Decade')
    plt.plot(x_axis, skills_scores, color='b')
    plt.axvline(x=1930, color='r')
    r_patch = mlines.Line2D([],[],color='r', label='Change Point')
    b_patch = mlines.Line2D([],[],color='b', label='Cosine Similarity')
    plt.legend(handles=[r_patch,b_patch])
    plt.savefig("skills.pdf")
