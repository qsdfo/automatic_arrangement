#!/usr/bin/env python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import csv as csv
import math

def read_csv_results(path):
    dico_hparam = {}
    # Read hparam into lists of values
    with open(path + '/result.csv', 'rb') as f:
        reader = csv.DictReader(f, delimiter=';')
        for r in reader:
            for hparam_name, hparam_value in r.iteritems():
                if hparam_name not in dico_hparam:
                    dico_hparam[hparam_name] = []
                dico_hparam[hparam_name].append(hparam_value)
    return dico_hparam

def plot_hparam(path):
    dico_hparam = read_csv_results(path)
    accuracy_list = dico_hparam['accuracy']

    avoid_hparam = ['accuracy', 'optim', 'ID', 'loss', 'model', 'lr']
    hparam_name_to_plot = [e for e in dico_hparam.keys() if e not in avoid_hparam]

    grid_size = int(math.ceil(math.sqrt(len(hparam_name_to_plot))))
    f, axarr = plt.subplots(grid_size, grid_size)
    counterx = 0
    countery = 0

    for hparam_name in hparam_name_to_plot:

        hparam_value = dico_hparam[hparam_name]

        # Four axes, returned as a 2-d array
        axarr[counterx, countery].plot(hparam_value, accuracy_list, 'bo')
        axarr[counterx, countery].set_title(hparam_name, fontsize=14, fontweight='bold')
        axarr[counterx, countery].set_xlabel('value of hparam')
        axarr[counterx, countery].set_ylabel('Accuracy score')

        counterx = counterx + 1
        if counterx == grid_size:
            counterx = 0
            countery = countery + 1

    f.subplots_adjust(hspace=0.58)
    plt.savefig(path + '/hparam_name.pdf')
    plt.close(f)


if __name__ == '__main__':
    plot_hparam("/home/aciditeam-leo/Aciditeam/lop/Results_guillimin/27_02_17/Results/event_level/discrete_units/quantization_100/gradient_descent/cRBM")
