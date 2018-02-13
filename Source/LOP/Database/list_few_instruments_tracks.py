#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import csv
import re
import os
import glob
import LOP.Scripts.config as config
import build_data_aux
import build_data_aux_no_piano


def list_tracks(folder_path):
    csv_files = glob.glob(folder_path + '/*.csv')
    csv_files_clean = [e for e in csv_files if not(re.search(u'metadata.csv', e)) and not(re.search(ur'.*_solo\.csv$', e))]
    if len(csv_files_clean) != 1:
        import pdb; pdb.set_trace()
    with open(csv_files_clean[0], 'rb') as ff:
        csvreader = csv.DictReader(ff, delimiter=";")
        row = csvreader.next()
    return len(set(row.values()))

if __name__ == '__main__':
    MIN_INSTRU = 4
    # Database have to be built jointly so that the ranges match
    DATABASE_PATH = config.database_root()
    DATABASE_NAMES = ["bouliane", "hand_picked_Spotify", "liszt_classical_archives", "imslp"]
    DATABASE_PATH_PRETRAINING = os.path.join(config.database_pretraining_root(), 'SOD')
    DATABASE_NAMES_PRETRAINING = ["Kunstderfuge", "Musicalion", "Mutopia", "OpenMusicScores"]

    # Create a list of paths
    def build_filepaths_list(db_path=DATABASE_PATH, db_names=DATABASE_NAMES):
        folder_paths = []
        for db_name in db_names:
            path = db_path + '/' + db_name
            for file_name in os.listdir(path):
                if file_name != '.DS_Store':
                    this_path = db_name + '/' + file_name
                    folder_paths.append(this_path)
        return folder_paths
    
    folder_paths = build_filepaths_list(DATABASE_PATH, DATABASE_NAMES)
    folder_paths = [os.path.join(DATABASE_PATH, e) for e in folder_paths]
    
    folder_paths_pretraining = build_filepaths_list(DATABASE_PATH_PRETRAINING, DATABASE_NAMES_PRETRAINING)
    folder_paths_pretraining = [os.path.join(DATABASE_PATH_PRETRAINING, e) for e in folder_paths_pretraining]

    rotten_files = []
    for track in (folder_paths_pretraining + folder_paths):
        num_instru = list_tracks(track)
        if num_instru < MIN_INSTRU:
            rotten_files.append(track)

    with open("few_instrument_files.txt", 'wb') as ff:
        for rotten_file in rotten_files:
            # Get only the last part
            split_filename = re.split('/', rotten_file)
            ff.write(split_filename[-2] + '/' + split_filename[-1] + '\n')
    