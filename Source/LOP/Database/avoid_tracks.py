#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import glob
import LOP.Scripts.config as config

def avoid_tracks():

	training_avoid = [
		# Too good
		# os.path.join(config.database_root(), "hand_picked_Spotify/40"),
		# os.path.join(config.database_root(), "hand_picked_Spotify/45"),
	] 
	
	# All IMSLP files
	# imslp_files = glob.glob(config.database_root() + '/imslp/[0-9]*')
	# training_avoid += imslp_files

	pre_training_avoid = [
		os.path.join(config.database_pretraining_root(), "Musicalion/1576"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3362"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3372"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3380"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3382"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3386"),
		os.path.join(config.database_pretraining_root(), "Kunstderfuge/1434"),
	]

	tracks_with_too_few_instruments = []
	with open("few_instrument_files_pretraining.txt", 'rb') as ff:
		for line in ff:
			tracks_with_too_few_instruments.append(os.path.join(config.database_pretraining_root(), line.rstrip("\n")))
	with open("few_instrument_files.txt", 'rb') as ff:
		for line in ff:
			tracks_with_too_few_instruments.append(os.path.join(config.database_root(), line.rstrip("\n")))
	
	# return training_avoid + pre_training_avoid + tracks_with_too_few_instruments
	return training_avoid + pre_training_avoid

def no_valid_tracks():
	no_valid_tracks = [
		# Too good
		os.path.join(config.database_root(), "hand_picked_Spotify/40"),
		os.path.join(config.database_root(), "hand_picked_Spotify/45"),
	] 
	return no_valid_tracks

if __name__ == "__main__":
	ret = avoid_tracks()
	import pdb; pdb.set_trace()