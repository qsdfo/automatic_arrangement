#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

from acidano.data_processing.midi.build_data import build_data

if __name__ == '__main__':
    import logging
    # log file
    log_file_path = 'log/main_log'
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

    # DATABASE_PATH = '/Users/leo/Recherche/GitHub_Aciditeam/database/Orchestration/LOP_database_29_05_17'
    DATABASE_PATH = '/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_29_05_17'
    INDEX_PATH = DATABASE_PATH + '/tvt_split'
    data_folder = '../Data'
    index_files_dict = {}
    index_files_dict['train'] = [
        # INDEX_PATH + "/debug_train.txt",
        INDEX_PATH + "/bouliane_train.txt",
        INDEX_PATH + "/hand_picked_Spotify_train.txt",
        INDEX_PATH + "/liszt_classical_archives_train.txt",
        INDEX_PATH + "/imslp_train.txt"
    ]
    index_files_dict['valid'] = [
        # INDEX_PATH + "/debug_valid.txt",
        INDEX_PATH + "/bouliane_valid.txt",
        INDEX_PATH + "/hand_picked_Spotify_valid.txt",
        INDEX_PATH + "/liszt_classical_archives_valid.txt",
        INDEX_PATH + "/imslp_valid.txt"
    ]
    index_files_dict['test'] = [
        # INDEX_PATH + "/debug_test.txt",
        INDEX_PATH + "/bouliane_test.txt",
        INDEX_PATH + "/hand_picked_Spotify_test.txt",
        INDEX_PATH + "/liszt_classical_archives_test.txt",
        INDEX_PATH + "/imslp_test.txt"
    ]

    # Dictionary with None if the data augmentation is not used, else the value for this data augmentation
    # Pitch translation. Write [0] for no translation
    max_translation = 3
    pitch_translations = range(-max_translation, max_translation+1)

    build_data(root_dir=DATABASE_PATH,
               index_files_dict=index_files_dict,
               meta_info_path=data_folder + '/temp.p',
               quantization=100,
               unit_type='binary',
               temporal_granularity='event_level',
               store_folder=data_folder,
               pitch_translation_augmentations=pitch_translations,
               logging=logging)
