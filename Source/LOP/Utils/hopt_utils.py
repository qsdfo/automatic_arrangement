#!/usr/bin/env python
# -*- coding: utf8 -*-

from LOP.Utils.hopt_wrapper import qloguniform_int
from hyperopt import hp
from math import log

def multi_layer_hopt(min_unit, max_unit, step, min_num_layer, max_num_layer, name):
    return hp.choice(name, [
        [qloguniform_int(name+'_'+str(i), log(min_unit), log(max_unit), step) for i in range(num_layer)] \
            for num_layer in range(min_num_layer, max_num_layer)
            ])