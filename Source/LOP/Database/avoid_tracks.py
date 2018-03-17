#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import glob
import LOP.Scripts.config as config

def avoid_tracks():

	training_avoid = [] 

	pre_training_avoid = [
		os.path.join(config.database_pretraining_root(), "Musicalion/1576"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3362"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3372"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3380"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3382"),
		os.path.join(config.database_pretraining_root(), "Musicalion/3386"),
		os.path.join(config.database_pretraining_root(), "Kunstderfuge/1434"),
	]
	
	# return training_avoid + pre_training_avoid + tracks_with_too_few_instruments
	return training_avoid + pre_training_avoid

def no_valid_tracks():
	no_valid_tracks = [
		# Too good
		os.path.join(config.database_root(), "hand_picked_Spotify/40"),
		os.path.join(config.database_root(), "hand_picked_Spotify/45"),
		os.path.join(config.database_root(), "imslp/21"),
		os.path.join(config.database_root(), "imslp/43"),
		os.path.join(config.database_root(), "imslp/20"),
		os.path.join(config.database_root(), "imslp/44"),
		os.path.join(config.database_root(), "imslp/22"),
		os.path.join(config.database_root(), "imslp/12"),
		os.path.join(config.database_root(), "imslp/14"),
		os.path.join(config.database_root(), "imslp/62"),
		os.path.join(config.database_root(), "imslp/68"),
		os.path.join(config.database_root(), "imslp/39"),
		os.path.join(config.database_root(), "imslp/15"),
		os.path.join(config.database_root(), "imslp/26"),
		os.path.join(config.database_root(), "imslp/71"),
		os.path.join(config.database_root(), "imslp/3"),
		os.path.join(config.database_root(), "imslp/78"),
		os.path.join(config.database_root(), "imslp/11"),
		os.path.join(config.database_root(), "imslp/86"),
		os.path.join(config.database_root(), "imslp/16"),
		os.path.join(config.database_root(), "imslp/25"),
		os.path.join(config.database_root(), "imslp/56"),
		os.path.join(config.database_root(), "imslp/77"),
		os.path.join(config.database_root(), "imslp/5"),
		os.path.join(config.database_root(), "imslp/23"),
		os.path.join(config.database_root(), "imslp/45"),
		os.path.join(config.database_root(), "imslp/50"),
		os.path.join(config.database_root(), "imslp/64"),
		os.path.join(config.database_root(), "debug/1"),
		os.path.join(config.database_root(), "debug/2"),
	] 

	# All IMSLP files
	# imslp_files = glob.glob(config.database_root() + '/imslp/[0-9]*')
	# training_avoid += imslp_files

	tracks_with_too_few_instruments = []
	# with open(config.data_root() + "/few_instrument_files_pretraining.txt", 'rb') as ff:
	# 	for line in ff:
	# 		tracks_with_too_few_instruments.append(os.path.join(config.database_pretraining_root(), line.rstrip("\n")))
	with open(config.data_root() + "/few_instrument_files.txt", 'rb') as ff:
		for line in ff:
			tracks_with_too_few_instruments.append(os.path.join(config.database_root(), line.rstrip("\n")))
	
	return no_valid_tracks + tracks_with_too_few_instruments

if __name__ == "__main__":
	ret = avoid_tracks()
	import pdb; pdb.set_trace()