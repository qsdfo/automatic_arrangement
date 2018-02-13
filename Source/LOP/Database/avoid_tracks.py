#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import glob
import LOP.Scripts.config as config

def avoid_tracks():
	# All IMSLP files
	imslp_files = glob.glob(config.database_root() + '/imslp/[0-9]*')

	training_avoid = [
		os.path.join(config.database_root(), "hand_picked_Spotify/40"),
		os.path.join(config.database_root(), "hand_picked_Spotify/45"),
	] + imslp_files

	pre_training_avoid = [
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/1576"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3362"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3372"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3380"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3382"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3386"),
		os.path.join(config.database_pretraining_root(), "SOD", "Kunstderfuge/1434"),
	]

	return training_avoid + pre_training_avoid

if __name__ == "__main__":
	ret = avoid_tracks()
	import pdb; pdb.set_trace()