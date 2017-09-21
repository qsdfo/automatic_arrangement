#!/usr/bin/env python
# -*- coding: utf8 -*-

def import_configs():
	configs= {
		'0': {
			'batch_size' : 200,
			'temporal_order' : 5,
			'dropout_probability' : 0,
			'weight_decay_coeff' : 0,
			'layers' : [500, 500],
			'threshold' : 0,
			'weighted_ce' : 0
		},
	}
	return configs