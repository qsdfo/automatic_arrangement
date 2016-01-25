# Main script for LOP
import csv

from Data_processing import load_data

# Select a model (path to the .py file)
from Models.Temporal_RBM.temporal_binary_rbm import train
model_path = 'Temporal_RBM'
temporal_granularity = 'frame_level'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
database = '../Data/data.p'

# Set hyperparameters (can be a grid)
config_file = 'config.csv'
result_file = '../Results/' + temporal_granularity + '/' + model_path + '/results.csv'
# Import hyperparams from a csv file (config.csv) and run each row in this csv
hyper_parameters = {}
with open(config_file, 'rb') as csvfile:
    config_csv = csv.reader(csvfile, delimiter=';')
    headers_config = config_csv.next()
    config_number = 0
    for row in config_csv:
        column = 0
        this_hyperparam = {}
        for hyperparam in headers_config:
            this_hyperparam[hyperparam]  = row[column]
            column += 1
        hyper_parameters[config_number] = this_hyperparam
        config_number += 1
# Import from result.csv the alreday tested configurations in a dictionnary
checked_config = {}
with open(result_file, 'rb') as csvfile:
    result_csv = csv.reader(csvfile, delimiter=';')
    headers_result = result_csv.next()
    result_number = 0
    for row in result_csv:
        column = 0
        this_hyperparam = {}
        for hyperparam in headers_config:  # /!\ Note that we use the header of the config file
            this_hyperparam[hyperparam]  = row[column]
            column += 1
        checked_config[result_number] = this_hyperparam
        config_number += 1

# Write the result in result.csv
# Compare granularity with granularity in the config_file

# Train the model, looping over the hyperparameters configurations
for config_hp in hyper_parameters.itervalues():
    # Before training for an hyperparam point, check if it has already been tested.
    #   If it's the case, values would be stored in an other CSV files (result.csv), with its performance
    NO_RUN = False
    for result_hp in checked_config.itervalues():
        if result_hp == config_hp:
            NO_RUN = True
            break
    if NO_RUN:
        log_file.write()
        break
    performance = train(config_hp, database, output_folder)
