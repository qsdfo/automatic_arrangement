#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import cPickle as pkl
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from acidano.utils.measure import accuracy_measure_not_shared, accuracy_measure_not_shared_continuous

DATA_DIR = '../Data'


def load_split_data(seq_length, set_identifier, data_folder):
    piano = np.load(data_folder + '/piano_' + set_identifier + '.csv')
    orch = np.load(data_folder + '/orchestra_' + set_identifier + '.csv')
    tracks_start_end = pkl.load(open(data_folder + '/tracks_start_end_' + set_identifier + '.pkl', 'rb'))
    X = []
    Y = []
    for (start, end) in tracks_start_end.values():
        for t in range(start+seq_length, end):
            # We predict t with t-seq_length to t-1
            X.append(np.concatenate((orch[t-seq_length:t].ravel(), piano[t])))
            Y.append(orch[t])
    return X, Y


# Param
temporal_order = 2
data_folder = DATA_DIR

# Load data
X_train, Y_train = load_split_data(temporal_order, 'train', data_folder)
X_valid, Y_valid = load_split_data(temporal_order, 'valid', data_folder)

# Train
clf = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
# clf = LabelPowerset(GaussianNB())
clf = clf.fit(X_train, Y_train)

# Predict
Y_predict = clf.predict(X_valid)
# Accuracy
accuracy = accuracy_measure_not_shared(Y_predict, Y_valid)
import pdb; pdb.set_trace()