#!/usr/bin/env python
# -*- coding: utf8 -*-

import re
import glob
import csv

def process_result_csv(path):
    config_paths = glob.glob(path + '/*')
    id_result_list = []
    # Walk through config folders
    for config_path in config_paths:
        # Get id
        ID = re.split(r'/', config_path)[-1]
        # Read result.csv
        result_file_path = config_path + '/result.csv'
        result_dict = {}
        with open(result_file_path, 'rb') as f:
            reader = csv.reader(f)
            for k, v in reader:
                result_dict[k] = v
        # Store results in a list (ID, accuracy, loss)
        id_result_list.append((ID, result_dict['accuracy'], result_dict['loss']))

    # Sort the list
    sorted_result = sorted(id_result_list, key=lambda x: x[1])

    # Write it in a csv file
    with open(path + '/result.csv', 'rb') as f:
        for elem in sorted_result:
            f.write(";".join(elem))
    return
