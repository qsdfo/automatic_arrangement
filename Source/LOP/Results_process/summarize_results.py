#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Scatter plot to find the study the correlation between accuracy and cross-entropy
Created on Thu Nov 30 17:07:52 2017

@author: leo
"""

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
import re
import shutil
import glob
import csv
import numpy as np

def main(config_folder, measure_name, avoid_folds):
    """Plots learning curves
    
    Paramaters
    ----------
    config_folders : str list
        list of strings containing the configurations from which we want to collect results
    name_measure_A : str
        name of the first measure (has to match a .npy file in the corresponding configuration folder)
    name_measure_B : str
        name of the second measure
        
    """

    plt.clf()

    res_summary = config_folder + '/result_summary'

    # csv file
    csv_file = os.path.join(res_summary, measure_name + '.csv')
    fieldnames = ['fold', 'short_range', 'short_range_best_epoch', 'long_range', 'long_range_best_epoch']
    with open(csv_file, 'wb') as ff: 
        writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

    # Plot file
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    plot_file = os.path.join(res_summary, measure_name + '.csv')

    # List folds
    folds = glob.glob(config_folder + '/[0-9]*')

    folds = [e for e in folds if re.split("/", e)[-1] not in avoid_folds]

    # Best means
    best_mean_short = []
    best_mean_long = []

    # Collect numpy arrays
    for fold in folds:
        fold_num = int(re.split('/', fold)[-1])

        results_long_range = np.loadtxt(fold + '/results_long_range/' + measure_name + '.txt')
        results_short_range = np.loadtxt(fold + '/results_short_range/' + measure_name + '.txt')

        # Avoid Nan ? Not recommended
        # if np.all(np.isnan(results_short_range)):
        #     continue


        with open(fold + '/results_long_range/' + measure_name + '_best_epoch.txt', 'rb') as ff:
            long_range_best = int(ff.read())
        with open(fold + '/results_short_range/' + measure_name + '_best_epoch.txt', 'rb') as ff:
            short_range_best = int(ff.read())

        #########################################################################################################
        #########################################################################################################
        #########################################################################################################
        # A changer dans le future
        # Hacky parce que j'ai oublie d'arreter le monitoring a la fin du training....
        # def remove_tail(mat, best):
        #     if len(np.nonzero(mat)[0])!=0:
        #         last_ind = max(np.nonzero(mat)[0][-1], best+1)
        #     else:
        #         last_ind = best+1
        #     return mat[:last_ind]
        # results_long_range = remove_tail(results_long_range, long_range_best)
        # results_short_range = remove_tail(results_short_range, short_range_best)
        #########################################################################################################
        #########################################################################################################
        #########################################################################################################
        plot_short, = plt.plot(results_short_range, color=colors[fold_num%len(colors)])
        plot_long, = plt.plot(results_long_range, color=colors[fold_num%len(colors)], ls='--', marker='o')

        if len(results_short_range.shape) > 0:
            best_mean_short.append(results_short_range[short_range_best])
            best_mean_long.append(results_long_range[long_range_best])
        else:
            best_mean_short = results_short_range
            best_mean_long = results_long_range

        # Write csv
        with open(csv_file, 'ab') as ff:
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            if len(results_short_range.shape) > 0:
                writer.writerow({'fold': fold_num,
                    'short_range': results_short_range[short_range_best],
                    'short_range_best_epoch' : short_range_best, 
                    'long_range' : results_long_range[long_range_best], 
                    'long_range_best_epoch': long_range_best})
            else:
                writer.writerow({'fold': fold_num,
                    'short_range': results_short_range,
                    'short_range_best_epoch' : 0, 
                    'long_range' : results_long_range, 
                    'long_range_best_epoch': 0})

    # Legend and plots
    # traits = mlines.Line2D([], [], color='black',
    #                       markersize=15, label='Short range task')
    # ronds = mlines.Line2D([], [], color='black', marker='o',
    #                       markersize=15, label='Long range task')
    # plt.legend(handles=[traits, ronds])
    plt.legend([plot_short, plot_long], ['short range prediction', 'long range prediction'])
    plt.title(measure_name + ' curves')
    plt.xlabel('epochs')
    plt.ylabel(measure_name)
    plt.savefig(res_summary + '/' + measure_name + ".pdf")

    # Inter-measure csv file
    # Exists ?
    all_mes_file = res_summary + '/all_measures_foldMean.csv'
    fieldnames = ['measure', 'short term', 'long term']
    if not os.path.isfile(all_mes_file):
        with open(all_mes_file, 'wb') as ff:
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
    with open(all_mes_file, 'ab') as ff:
        writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
        writer.writerow({'measure': measure_name,
            'short term': np.mean(best_mean_short),
            'long term': np.mean(best_mean_long)})
    return

if __name__ == '__main__':
    config_folders = ["/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/NADE/AA"]
    # aaa="/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/Measure/qualitative_evaluation_different_training_criterion/fully_trained/Xent_tn_fuly_trained"
    # bbb="/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/Measure/qualitative_evaluation_different_training_criterion/fully_trained/Xent_tn_static_bias_fully_trained"
    # config_folders = [
    #     aaa + "/30",
    #     aaa + "/31",
    #     aaa + "/32",
    #     aaa + "/33",
    #     aaa + "/34",
    #     aaa + "/35",
    #     aaa + "/36",
    #     aaa + "/37",
    #     bbb + "/300",
    #     bbb + "/301",
    #     bbb + "/302",
    #     bbb + "/303",
    #     bbb + "/304",
    #     bbb + "/305",
    #     bbb + "/306",
    #     bbb + "/307",
    #     ]

    measures = ["accuracy", "loss", "f_score", "precision", "recall", "true_accuracy", "Xent"]

    avoid_folds = ["2", "14"] # Should be set to [] most of the times

    for config_folder in config_folders:    
        
        if not os.path.isdir(config_folder):
            continue
        
        # Create store folder
        res_summary = config_folder + '/result_summary'
        if os.path.isdir(res_summary):
            shutil.rmtree(res_summary)
        os.mkdir(res_summary)

        for measure in measures:
            main(config_folder, measure, avoid_folds)