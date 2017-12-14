#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:58:17 2017

@author: leo
"""


import glob
import re


"""
Shitty script parce que j'ai été trop con pour penser à écrire la best epoch dans result.csv...
heureusement c'est dans log, mais c'est relou
"""

if __name__ == '__main__':
    # Collect the data
    root = "/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Experiments/Fixed_static_biases/precomputed_fixed_static_biases_quanti" +\
        "/LSTM_static_bias"
    configs = glob.glob(root + '/[0-9]')
    
    for config in configs:
        # Read log and extract number poch
        epoch = []
        with open(config + '/log.txt', 'rb') as ff:
            for line in ff:
                m = re.search(r"Best model obtained at epoch :  ([0-9]+)", line)
                if m:
                    epoch.append(m.group(1))
        with open(config + '/best_epoch.txt', 'wb') as ff:
            for ll in epoch:
                ff.write(ll + '\n')
        