#!/usr/bin/env python
# -*- coding: utf8 -*-

def import_configs():
	configs= {
		'0': {
			'batch_size' : 200,
			'temporal_order' : 5,
			'dropout_probability' : 0,
			'weight_decay_coeff' : 0,
			'num_filter_piano': 20,
			'kernel_size_piano': 12,
			'mlp_piano': [500, 500],
			'mlp_pred': [500, 500],
			'gru_orch': [500, 500],
		},
	}
	return configs