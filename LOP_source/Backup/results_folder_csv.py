#!/usr/bin/env python
# -*- coding: utf8 -*-

import re
import csv
import cPickle as pkl

def get_results_and_id(config_path):
    # Given the path for a configuration,
    # simply returns a tuple containing the ID, accuracy and loss

    # Get id
    ID = re.split(r'/', config_path)[-1]
    # Read result.csv
    result_file_path = config_path + '/result.csv'
    result_dict = {}
    with open(result_file_path, 'rb') as f:
        for row in f:
            key, value = re.split(';', row)
            result_dict[key] = value.strip('\n')
    # Store results in a list (ID, accuracy, loss)
    return (ID, float(result_dict['accuracy']), float(result_dict['loss']))

def write_csv_results(path, id_result_list, first_config):
    # Sort the results list according to accuracy
    sorted_result = sorted(id_result_list, key=lambda x: -x[1])

    # Get header from the model of the first configuration
    space_path = first_config + '/config.pkl'
    space = pkl.load(open(space_path, 'rb'))
    headers = ['ID', 'accuracy', 'loss', 'model']
    headers_model = (space['model'].keys())
    headers_optim = (space['optim'].keys())
    headers = headers + headers_model + ['optim'] + headers_optim
    del(space)

    # Write it in a csv file
    with open(path + '/result.csv', 'wb') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=headers)
        writer.writeheader()
        for elem in sorted_result:
            # Get ID
            ID = elem[0]
            # Get config
            config_folder = path + '/' + ID
            param_path = config_folder + '/config.pkl'
            space = pkl.load(open(param_path, 'rb'))
            model_param = space['model']
            model_name = space['script']['model_class']
            optim_param = space['optim']
            optim_name = space['script']['optimization_method']
            # Write result as a dict
            result_dict = {'ID': elem[0], 'accuracy': elem[1], 'loss': elem[2], 'model': model_name, 'optim': optim_name}
            result_dict.update(model_param)
            result_dict.update(optim_param)
            # Write
            writer.writerow(result_dict)
    return
