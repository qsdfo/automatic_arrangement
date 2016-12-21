#!/usr/bin/env python
# -*- coding: utf8 -*-

import re
import glob
import csv
import cPickle as pkl

def process_result_csv(path):
    config_paths = glob.glob(path + '/*')
    id_result_list = []
    # Walk through config folders
    import pdb; pdb.set_trace()
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

    # Get header from the model of the first configuration
    model_path = config_path[0] + '/model.pkl'
    model = pkl.load(open(model_path, 'rb'))
    qzdqzde = model.get_hp_space().keys()
    import pdb; pdb.set_trace()

    # headers = ['ID', 'accuracy', 'loss'] + model.get_hp_space().keys()
    del(model)

    # Write it in a csv file
    with open(path + '/result.csv', 'wb') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=headers)
        writer.writerow(headers)
        for elem in sorted_result:
            # Get ID
            ID = elem[0]
            # Get config
            config_folder = path + '/' + ID
            param_path = config_folder + '/config.pkl'
            space = pkl.load(open(param_path, 'rb'))
            model_param = space['model']
            # Write result as a dict
            result_dict = {'ID': elem[0], 'accuracy': elem[1], 'loss': elem[2]}
            result_dict.update(model_param)
            # Write
            writer.writerow(result_dict)
    return

if __name__ == '__main__':
    process_result_csv('/home/aciditeam-leo/Aciditeam/lop/Results/event_level/discrete_units/quantization_4/gradient_descent/LSTM/')
