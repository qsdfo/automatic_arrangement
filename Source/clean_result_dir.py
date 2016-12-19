#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import os

def clean(path):
    list_dir = glob.glob(path + '/*')
    for dirname in list_dir:
        if os.listdir(dirname) == []:
            os.rmdir(dirname)

if __name__ == '__main__':
    clean('/home/aciditeam-leo/Aciditeam/lop/Results/event_level/discrete_units/quantization_4/gradient_descent/LSTM/Grid_search')
