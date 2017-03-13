#!/usr/bin/env python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import csv as csv
import math
import numpy as np

from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool


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
    accuracy_list = np.asarray([float(e) for e in dico_hparam['accuracy']])

    # Color list
    min_red = 38
    min_green = 87
    min_blue = 235
    max_red = 222
    max_green = 97
    max_blue = 96
    scaled_accuracy_list_blue = [(min_blue + (max_blue-min_blue)*e/max(accuracy_list)) for e in accuracy_list]
    scaled_accuracy_list_green = [(min_green + (max_green-min_green)*e/max(accuracy_list)) for e in accuracy_list]
    scaled_accuracy_list_red = [(min_red + (max_red-min_red)*e/max(accuracy_list)) for e in accuracy_list]
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b)) for r,g,b in zip(scaled_accuracy_list_blue, scaled_accuracy_list_green, scaled_accuracy_list_red)
    ]

    avoid_hparam = ['accuracy', 'optim', 'ID', 'loss', 'model', 'lr']
    dico_hparam_to_plot = {e: dico_hparam[e] for e in dico_hparam.keys() if e not in avoid_hparam}
    source = ColumnDataSource(
            data=dico_hparam
        )

    list_plot = []
    output_file(path + "/hparam.html")

    for hparam_name in dico_hparam_to_plot.keys():
        hover = HoverTool(tooltips=[(e, "@" + e) for e in dico_hparam.keys()])
        s = figure(title=hparam_name)
        s.add_tools(hover)
        s.xaxis.axis_label = hparam_name
        s.yaxis.axis_label = "Accuracy"
        s.circle(hparam_name, "accuracy", size=10, color=colors, source=source)
        list_plot.append(s)
    # make a grid
    grid = gridplot(list_plot, ncols=2, plot_width=250, plot_height=250)
    save(grid)

if __name__ == "__main__":
    plot_hparam("/home/aciditeam-leo/Aciditeam/lop/Results_guillimin/27_02_17/Results/event_level/binary/quantization_100/gradient_descent/RBM_inpainting/")
