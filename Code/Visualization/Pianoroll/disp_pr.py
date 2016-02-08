#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Script which open the proper html/D3.js file given a name and instrument
import sys
import shutil
from sys import platform as _platform
from subprocess import call

path_to_csv = sys.argv[1]  # Whole path to the CSV file containing the data
local_file = 'data.csv'
shutil.copy(path_to_csv, local_file)
if _platform == "linux" or _platform == "linux2":
    call(["firefox", "pianoroll.html"])
elif _platform == "darwin":
    call(["open", "pianoroll.html"])
