#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import glob
import re

result_file = 'result.csv'
if os.path.isfile(result_file):
    os.remove(result_file)

def read_line_logtxt(line):
    res = re.split('=', line)
    if len(res) != 2:
        return None, None
    else:
        return res[0].strip(), res[1].strip()

if __name__ =='__main__':
    list_dirs = glob.glob("*/")
    for dir_path in list_dirs:
        dic = {}
        with open(dir_path + '/log.txt', 'rb') as f:
            # Read file in dic
            for line in f:
                key, value = read_line_logtxt(line)
                if key and value:
                    dic[key] = value
        # Write dic in csv file
        if not os.path.isfile(result_file):
            # Write header (first time)
            with open(result_file, 'wb') as f:
                string = ";".join(dic.keys())
                f.write(string + "\n")
        with open(result_file, 'ab') as f:
            string = ";".join(dic.values())
            f.write(string + "\n")
