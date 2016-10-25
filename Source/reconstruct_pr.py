#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv


def reconstruct_pr(matrix, mapping):
    # Reconstruct an orchestral dictionnary pianoroll from
    #   - matrix : (time * pitch)
    #   - mapping : a dictionnary mapping index space of the matrix
    #               to the pitch space of each instrument

    pr_instru = {}

    max_velocity = 127
    # Dimensions of each instrument pianoroll
    T = matrix.shape[0]
    N = 128

    for instrument_name, ranges in mapping.iteritems():
        index_min = ranges['index_min']
        index_max = ranges['index_max']
        pitch_min = ranges['pitch_min']
        pitch_max = ranges['pitch_max']

        this_pr = np.zeros((T,N), dtype=np.int16)
        this_pr[:,pitch_min:pitch_max] = matrix[:,index_min:index_max]
        this_pr = this_pr * max_velocity
        pr_instru[instrument_name] = this_pr

    return pr_instru


if __name__ == '__main__':
    import cPickle as pickle
    metadata = pickle.load(open('../Data/metadata.pkl', 'rb'))
    instru_mapping = metadata['instru_mapping']
    pr = np.tile(np.arange(1,590,1), (50,1))
    pr_instru = reconstruct_pr(pr, instru_mapping, False)

    # Visualisation
    AAA = np.concatenate(pr_instru.values(), axis=1)
    temp_csv = 'temp.csv'
    np.savetxt(temp_csv, AAA, delimiter=',')
    dump_to_csv(temp_csv, temp_csv)
    write_numpy_array_html("temp.html", "temp")
