import numpy as np
from collections import Counter

PhysioProp = {'Hydrophobicity': {'1': ['R', 'K', 'E', 'D', 'Q', 'N'], '2': ['G', 'A', 'S', 'T', 'P', 'H', 'Y'],
                                 '3': ['C', 'L', 'V', 'I', 'M', 'F', 'W']}, \
              'NormalizedVolume': {'1': ['G', 'A', 'S', 'T', 'P', 'D', 'C'], '2': ['N', 'V', 'E', 'Q', 'I', 'L'],
                                   '3': ['M', 'H', 'K', 'F', 'R', 'Y', 'W']}, \
              'Polarity': {'1': ['L', 'I', 'F', 'W', 'C', 'M', 'V', 'Y'], '2': ['P', 'A', 'T', 'G', 'S'],
                           '3': ['H', 'Q', 'R', 'K', 'N', 'E', 'D']}, \
              'Polarizability': {'1': ['G', 'A', 'S', 'D', 'T'], '2': ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'],
                                 '3': ['K', 'M', 'H', 'F', 'R', 'Y', 'W']},
              'Charge': {'1': ['K', 'R'],
                         '2': ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
                         '3': ['D', 'E']}, \
              'SecondaryStructure': {'1': ['E', 'A', 'L', 'M', 'Q', 'K', 'R', 'H'],
                                     '2': ['V', 'I', 'Y', 'C', 'W', 'F', 'T'], '3': ['G', 'N', 'P', 'S', 'D']}, \
              'SolventAccessibility': {'1': ['A', 'L', 'F', 'C', 'G', 'I', 'V', 'W'],
                                       '2': ['P', 'K', 'Q', 'E', 'N', 'D'], '3': ['M', 'R', 'S', 'T', 'H', 'Y']}}


def PD_comp(seq_mod):
    c = Counter(seq_mod)
    pd_comp = np.zeros((1, 3))
    pd_comp[:, 0] = c[1] / seq_len
    pd_comp[:, 1] = c[2] / seq_len
    pd_comp[:, 2] = c[3] / seq_len
    return pd_comp


def PD_tran(seq_mod):
    p1 = seq_mod[::2]
    p2 = seq_mod[1::2]
    p1_ = [str(i) + str(j) for (i, j) in zip(p1, p2)]
    p2_ = [str(i) + str(j) for (i, j) in zip(p2, p1)]
    p = p1_ + p2_
    c = Counter(p)
    pd_trans = np.zeros((1, 3))
    pd_trans[:, 0] = (c['12'] + c['21']) / (seq_len - 1)
    pd_trans[:, 1] = (c['13'] + c['31']) / (seq_len - 1)
    pd_trans[:, 2] = (c['23'] + c['32']) / (seq_len - 1)
    return pd_trans


def PD_dist(seq_mod):
    t = [(i, j) for (i, j) in zip(range(1, seq_len + 1, 1), seq_mod)]
    p1 = [x[0] for x in t if x[1] == 1]
    p2 = [x[0] for x in t if x[1] == 2]
    p3 = [x[0] for x in t if x[1] == 3]
    p = [p1, p2, p3]
    pd_dist = np.zeros((1, 15))

    x = 0
    for i in range(3):
        if not p[i]:
            # how to deal with empty list
            x += 5
            continue
        pd_dist[:, x] = p[i][0] / seq_len
        pd_dist[:, x + 1] = find_index(p[i], 0.25) / seq_len
        pd_dist[:, x + 2] = find_index(p[i], 0.50) / seq_len
        pd_dist[:, x + 3] = find_index(p[i], 0.75) / seq_len
        pd_dist[:, x + 4] = p[i][-1] / seq_len
        x += 5
    return pd_dist


def find_index(l, fraction):
    llen = len(l)
    idx = int(llen * fraction)
    return l[idx]


combs = [str(i) + str(j) + str(k) for i in [1, 2, 3] for j in [1, 2, 3] for k in [1, 2, 3]]


def PD_freq(seq_mod):
    s = map(str, seq_mod)
    s = ''.join(s)
    p1 = [s[i:i + 3] for i in range(0, len(s), 3)]
    p2 = [s[i:i + 3] for i in range(1, len(s), 3)]
    p3 = [s[i:i + 3] for i in range(2, len(s), 3)]
    p = p1 + p2 + p3
    c = Counter(p)
    pd_freq = np.zeros((1, 27))
    for i in range(27):
        pd_freq[:, i] = c[combs[i]] / (seq_len / 3.)
    return pd_freq


def get_features(seq, PhysioPropSelection):
    PhysioFeats = np.empty((1, 1))
    if PhysioPropSelection =='all':
        PhysioPropSelection=PhysioProp.keys()
    global seq_len
    seq_len = len(seq)
    for key in PhysioPropSelection:
        seq_mod = [1 if x in PhysioProp[key]['1'] else x for x in seq]
        seq_mod = [2 if x in PhysioProp[key]['2'] else x for x in seq_mod]
        seq_mod = [3 if x in PhysioProp[key]['3'] else x for x in seq_mod]
        PhysioFeats = np.concatenate((PhysioFeats, PD_comp(seq_mod), PD_tran(seq_mod), PD_dist(seq_mod), PD_freq(seq_mod)), axis=1)
    PhysioFeats = PhysioFeats[:, 1:]
    return PhysioFeats

























