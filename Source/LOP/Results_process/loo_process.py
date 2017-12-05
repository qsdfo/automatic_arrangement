#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Process leave-one-out results
Created on Mon Nov 27 16:13:32 2017

@author: leo
"""

import csv
import os 
import re


def process(config_path, result_file):
    fieldnames = ['valid_file_name', 'accuracy', 'Xent']
    
    # Touch file
    with open(result_file, 'wb') as ff:    
        writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        
    for config_path in os.listdir(configs_path):
        if not re.search(r'^[0-9]+$', config_path):
            continue
        
        # Get scores
        result_csv_path = os.path.join(configs_path, config_path, 'result.csv')
        with open(result_csv_path, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                Xent = row['loss']
                Acc = row['accuracy']

        # Get name of the valid file
        valid_file_path = os.path.join(configs_path, config_path, 'valid_names.txt')
        with open(valid_file_path, 'rb') as ff:
            valid_file = ff.read()

        # Write everything in a csv
        with open(result_file, 'ab') as ff:
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            writer.writerow({'valid_file_name': valid_file, 'accuracy': Acc, 'Xent': Xent})

if __name__ == '__main__':
    configs_path = "/Users/leo/Recherche/GitHub_Aciditeam/lop/Results/LOO_Lstm"
    result_file = os.path.join(configs_path, 'acc_sorted.csv')
    process(configs_path, result_file)
    