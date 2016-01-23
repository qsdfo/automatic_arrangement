#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script which open the proper html/D3.js file given a name and instrument
import sys
import shutil
from subprocess import call

path_to_csv = sys.argv[1]  # Whole path to the CSV file containing the data
local_file = 'data.csv'
shutil.copy(path_to_csv, local_file)
call(["firefox", "pianoroll.html"])
