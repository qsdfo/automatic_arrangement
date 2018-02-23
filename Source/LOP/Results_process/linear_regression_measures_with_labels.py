#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Scatter plot to find the study the correlation between accuracy and cross-entropy
Created on Thu Nov 30 17:07:52 2017

@author: leo
"""

from matplotlib import pyplot as plt
import os
import shutil
import numpy as np
from LOP_database.visualization.numpy_array.visualize_numpy import visualize_mat_proba


def plot_preds_truth(source_folder):
	plot_folder = os.path.join(source_folder, 'plots')
	os.mkdir(plot_folder)
	# Load data
	preds = np.load(os.path.join(source_folder, 'preds.npy'))
	truth = np.load(os.path.join(source_folder, 'truth.npy'))
	for i in range(len(preds)):
		this_folder = os.path.join(plot_folder, str(i))
		temp_mat = np.stack((truth[i], preds[i]))
		visualize_mat_proba(temp_mat, this_folder, "YO", threshold=0.05)
	
	
def compare(x, name_measure_A, y, name_measure_B, result_folder):
	"""Compute the linear regression, scatter plot the points in a plan along with the curve derived from the linear reg.
	
	Parameters
	----------
	x : 1D npy array
		values corresponding to the first measure
	y : 1D npy array
		values corresponding to the second measure
	
	"""
	plt.clf()

	fig,ax = plt.subplots()
	sc = plt.scatter(x, y, c="b", alpha=0.5)
	plt.xlabel(name_measure_A)
	plt.ylabel(name_measure_B)
	
	annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
					bbox=dict(boxstyle="round", fc="w"),
					arrowprops=dict(arrowstyle="->"))
	annot.set_visible(False)
	
	def update_annot(ind):
		pos = sc.get_offsets()[ind["ind"][0]]
		annot.xy = pos
		text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
							   " ".join([str(n) for n in ind["ind"]]))
		annot.set_text(text)
		annot.get_bbox_patch().set_alpha(0.4)
	
	
	def hover(event):
		vis = annot.get_visible()
		if event.inaxes == ax:
			cont, ind = sc.contains(event)
			if cont:
				update_annot(ind)
				annot.set_visible(True)
				fig.canvas.draw_idle()
			else:
				if vis:
					annot.set_visible(False)
					fig.canvas.draw_idle()
	
	fig.canvas.mpl_connect("motion_notify_event", hover)
	
	plt.show()
	return

def process_measure(config_folders, name_measure_A, name_measure_B, result_folder):
	"""Gather the data from different configuration folders
	
	Paramaters
	----------
	config_folders : str list
		list of strings containing the configurations from which we want to collect results
	name_measure_A : str
		name of the first measure (has to match a .npy file in the corresponding configuration folder)
	name_measure_B : str
		name of the second measure
		
	"""

	measure_A_list = []
	measure_B_list = []
	for config_folder in config_folders:
		# Read npy
		measure_A_list.append(np.load(os.path.join(config_folder, name_measure_A + '.npy')))
		measure_B_list.append(np.load(os.path.join(config_folder, name_measure_B + '.npy')))
	measure_A = np.concatenate(measure_A_list)
	measure_B = np.concatenate(measure_B_list)
	compare(measure_A, name_measure_A, measure_B, name_measure_B, result_folder)
	return

if __name__ == '__main__':
	# source_folder = '/Users/leo/Recherche/GitHub_Aciditeam/lop/Results/MEASURE/scatter_measure_link_labels'
	# config_folders = [os.path.join(source_folder, str(0), 'debug', 'Xent_criterion')]
	config_folders = ["/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/Measure/qualitative_evaluation_different_training_criterion/fully_trained/Xent_tn_fuly_trained/30_bis/0"]
	result_folder = 'compare_measure'

	if os.path.isdir(result_folder):
		shutil.rmtree(result_folder)
	os.mkdir(result_folder)

	# plot_preds_truth(config_folders[0])
	process_measure(config_folders, "accuracy", "val_loss", result_folder)