import numpy as np
from collections import Counter
import warnings
warnings.simplefilter("error", RuntimeWarning)
import pickle
with open('alphabet.txt', 'rb') as fp: alphabet=pickle.load(fp)
#mean_comp = np.load('mean_comp.npy')
mean_comp = np.load('mean_comp_blind.npy')

def counts(seq):
    c = Counter(seq)
    cnts = np.zeros((1, 20))
    for idx, letter in enumerate(alphabet):
        cnts[:, idx] = c[letter]
    cnts[cnts == 1] = 2 # take care of potential edge case
    cnts[cnts == 0] = 2
    return cnts


def carr_type1(seq):
    seq_counter = Counter(seq)
    feats = np.zeros((1, 20))
    var = mean_comp * (1 - mean_comp) / seq_len
    for idx, letter in enumerate(alphabet):
        feats[:, idx] = seq_counter[letter] / seq_len
    feats = (feats - mean_comp) / (var ** 0.5)
    return feats


def carr_type2(seq, cent_pos_mean):
    theo_mean = (seq_len + 1) / 2.
    theo_var = (seq_len + 1) * (seq_len - seq_counts) / (12 * seq_counts)
    return (cent_pos_mean - theo_mean) / (theo_var ** 0.5)


def carr_type3(seq, cent_pos_mean):
    pos_vector = np.arange(1, len(seq) + 1, 1)
    distributional = np.zeros((1, 20))

    for idx, letter in enumerate(alphabet):
        pos = [pos_vector[idx] for idx, aa in enumerate(seq) if aa == letter]
        if pos:
            pos = np.asarray(pos, dtype=np.float32)
            distributional[:, idx] = np.sum((pos - cent_pos_mean[:, idx]) ** 2)
    measure_factor = (seq_len + 1) / (seq_len * (seq_counts - 1))
    measure = measure_factor * distributional
    theo_mean = (seq_len ** 2 - 1) / 12.
    theo_var = (seq_len - seq_counts) * ((seq_len - 1) ** 2) * (seq_len + 1) * (
    2 * seq_counts * seq_len + 3 * seq_len + 3 * seq_counts + 3) / (360 * seq_counts * (seq_counts - 1) * seq_len)
    return (measure - theo_mean) / (theo_var ** 0.5)


def cent_mean(seq):
    pos_vector = np.arange(1, len(seq) + 1, 1)
    centroidal_position = np.zeros((1, 20))
    for idx, letter in enumerate(alphabet):
        pos = [pos_vector[idx] for idx, aa in enumerate(seq) if aa == letter]
        try:
            centroidal_position[:, idx] = np.mean(pos)
        except:
            centroidal_position[:, idx] = 0
    return centroidal_position

def get_features(seq):
    global seq_len
    seq_len = len(seq)
    global seq_counts
    seq_counts=counts(seq)
    cent_pos_mean = cent_mean(seq)
    feats0 = seq_counts/seq_len
    feats1 = carr_type1(seq)
    feats2 = carr_type2(seq, cent_pos_mean)
    feats3 = carr_type3(seq, cent_pos_mean)
    return np.concatenate((feats0, feats1, feats2, feats3), axis=1)
