# Main script for LOP

# Select a model (path to the .py file)
from Models.RBM.temporal_binary_rbm import train
model_path = 'RBM'

# Build the matrix database (stored in a data.p file in Data) from a music XML database
database = '../Data/data.p'

# Output folder (where the results are saved)
output_path =

# Set hyperparameters (can be a grid)
config_file = 'config.csv'
result_file = '../'
# Import hyperparams from a csv file (config.csv) and run each row in this csv
# Before training for an hyperparam point, check if it has already been tested.
#               If it's the case, values would be stored in an other CSV files (result.csv), with its performance
# Write the result in result.csv

# Train the model
performance = train(hyper_parameters, database, output_folder)
