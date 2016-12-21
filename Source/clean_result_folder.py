#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import os
import shutil

def clean(path):
    list_dir = glob.glob(path + '/*')
    for dirname in list_dir:
        list_file = os.listdir(dirname)
        NO_RESULT_FILE = 'result.csv' not in list_file
        NO_CONFIG_FILE = 'config.pkl' not in list_file
        NO_MODEL_FILE = 'model.pkl' not in list_file
        if NO_CONFIG_FILE or NO_RESULT_FILE or NO_MODEL_FILE:
            shutil.rmtree(dirname)

if __name__ == '__main__':
    clean('/home/aciditeam-leo/Aciditeam/lop/Results/event_level/discrete_units/quantization_4/gradient_descent/LSTM')
