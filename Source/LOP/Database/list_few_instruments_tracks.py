#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import csv
import build_data_aux
import build_data_aux_no_piano


def list_tracks(folder_path):
    csv_file = glob.glob(folder_path + '*.csv')
    with open(csv_file, 'rb') as ff:
        csvreader = csv.DictReader(ff, delimiter=";")
        row = csvreader.next()
    return len(set(row.values()))

if __name__ == '__main__':
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

    for track in (folder_paths_pretraining + folder_paths):
        num_instru = list_tracks(track)
        if num_instru < 4:
            print(track)

    