#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import os
import shutil

def clean(path):
    list_dir = glob.glob(path + '/*')
    for dirname in list_dir:
        if os.path.isfile(dirname):
            continue
        list_file = os.listdir(dirname)
        # Minimum is a result file and a saved model
        RESULT_FILE = 'result.csv' in list_file
        MODEL_FILE = 'model' in list_file
        
        if not(RESULT_FILE and MODEL_FILE):
            shutil.rmtree(dirname)

if __name__ == '__main__':
    clean('/home/mil/leo/lop/Results/Data__event_level8__3/LSTM_plugged_base')
