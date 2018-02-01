#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:09:58 2017

@author: leo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:34:41 2017

@author: leo
"""

import os
import csv
import glob
import numpy as np

def compute_mean_config(config):
    folds = glob.glob(os.path.join(config, '[0-9]*'))
    Xent = []
    acc = []
    true_acc = []
    for fold in folds:
        # Read CSV
        with open(os.path.join(fold, 'result.csv')) as ff:
            reader = csv.DictReader(ff, delimiter=';')
            elem = reader.next()
            this_Xent = float(elem["Xent"])
            this_acc = float(elem["accuracy"])
            Xent.append(this_Xent)
            acc.append(this_acc)
    print("Acc : {:.3f} ; Xent : {:.3f}\n".format(np.mean(acc), np.mean(Xent)))
    return

if __name__ == '__main__':
    # Collect the data
    configs = ["/Users/leo/Recherche/GitHub_Aciditeam/automatic_arrangement/Results/Data_tempGran8/Baseline_Random/0"]
    
    for config in configs:
        compute_mean_config(config)