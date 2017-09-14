#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import glob

import matplotlib.pyplot as plt
import numpy as np

# import plotly.plotly as py
# import plotly.graph_objs as go
# py.sign_in('username', 'api_key')


def plot_intensity_distribution(path_data):

    # Load orchestral numpy arrays
    npy_files = glob.glob(path_data + "/orch*.npy")
    prs = [np.load(e) for e in npy_files]
    pr = np.concatenate(prs, axis=0)

    for pitch in range(pr.shape[1]):
        if pitch % 1 == 0:
            plt.hist(pr[:, pitch], 2)
            # plt.plot(bin_edges[:-1], hist[:])
            plt.savefig('DEBUG/intensity_distrib_' + str(pitch) + '.pdf')
            plt.clf()

if __name__ == '__main__':
    plot_intensity_distribution('/Users/leo/Recherche/GitHub_Aciditeam/lop/Data/Data__event_level100__0')