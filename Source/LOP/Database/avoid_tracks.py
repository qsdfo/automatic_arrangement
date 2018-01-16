#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import LOP.Scripts.config as config

def avoid_tracks():
	return [
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/1576"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3362"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3372"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3380"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3382"),
		os.path.join(config.database_pretraining_root(), "SOD", "Musicalion/3386"),
		os.path.join(config.database_pretraining_root(), "SOD", "Kunstderfuge/1434"),
	]