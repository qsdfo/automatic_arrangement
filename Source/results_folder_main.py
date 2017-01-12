#!/usr/bin/env python
# -*- coding: utf8 -*-


import glob
import logging

from results_folder_generate import generate_midi
from results_folder_plot_weights import plot_weights
from results_folder_csv import get_results_and_id, write_csv_results

# Main script for processing results folders
# Processing consists in :
#   per folder processing :
#       - generating midi
#       - plot model weights
#   "global" processing
#       - write result.csv files, which sums up and sort the results of the different hyper-parameter configurations
#       - plot hyper-parameter statistics


def processing_results(path, generation_length=50, seed_size=20, quantization_write=None):
    # path has to be the roots of the results folders
    # e.g. : /home/aciditeam-leo/Aciditeam/lop/Results/event_level/discrete_units/quantization_4/gradient_descent/LSTM

    ############################################################
    # Logging
    ############################################################
    log_file_path = 'generate_log'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger_generate = logging.getLogger('generate')
    logger_plot = logging.getLogger('plot_weights')

    ############################################################
    # Processing
    ############################################################
    # Get folders starting with a number
    configurations = glob.glob(path + '/[0-9]*')
    id_result_list = []
    for configuration in configurations:
        import pdb; pdb.set_trace()
        generate_midi(configuration, generation_length, seed_size, quantization_write, logger_generate)
        import pdb; pdb.set_trace()
        plot_weights(configuration, logger_plot)
        import pdb; pdb.set_trace()
        id_result_list.append(get_results_and_id(configuration))

    write_csv_results(id_result_list, configurations[0])

if __name__ == '__main__':
    processing_results('/home/aciditeam-leo/Aciditeam/lop/Results_bis/event_level/discrete_units/quantization_4/gradient_descent/RBM_inpainting')
