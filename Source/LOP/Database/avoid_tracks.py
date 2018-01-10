#!/usr/bin/env python
# -*- coding: utf-8-unix -*-


import os
import LOP.Scripts.config as config

def avoid_tracks():
	return [
		os.path.join(config.database_pretraining_root(), "SOD", "Kunstderfuge/1434")
	]