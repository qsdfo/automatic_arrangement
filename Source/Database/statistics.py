import csv
import build_data_aux
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim


DATABASE_PATH = "/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_29_05_17"

def get_number_of_file_per_composer(files):
    
    dict_orch = {}
    dict_piano = {}
    T_orch = 0
    T_piano = 0
    import pdb; pdb.set_trace()
    for file in files:
        with open(file, 'rb') as csvfile:        
            spamreader = csv.DictReader(csvfile, delimiter=';', fieldnames=["id","path","orch_composer","orch_song","orch_audio_path","solo_composer","solo_song","solo_audio_path"])
            # Avoid header
            spamreader.next()    
            for row in spamreader:        
                # Name and path
                orch_composer = row['orch_composer']
                piano_composer = row['solo_composer']
                path = DATABASE_PATH + '/' + row['path']
                # Read midi files
                try:
                    pianoroll_0, instru_0, _, _, pianoroll_1, instru_1, _, _ = build_data_aux.get_instru_and_pr_from_folder_path(path, 100)
                except:
                    import pdb; pdb.set_trace()
                    continue
                pr_piano, _, _, _, _,\
                pr_orch, _, _, _, _,\
                _ =\
                build_data_aux.discriminate_between_piano_and_orchestra(pianoroll_0, 0, 0, instru_0, 0,
                                                         pianoroll_1, 0, 0, instru_1, 0,
                                                         0)

                length_piano = len(sum_along_instru_dim(pr_piano))
                length_orch = len(sum_along_instru_dim(pr_orch))

                if orch_composer in dict_orch.keys():
                    dict_orch[orch_composer]['num_file'] += 1
                    dict_orch[orch_composer]['relative_length'] += length_orch
                else:
                    dict_orch[orch_composer] = {}
                    dict_orch[orch_composer]['num_file'] = 1
                    dict_orch[orch_composer]['relative_length'] = length_orch
                T_orch += length_orch
                    
                if piano_composer in dict_piano.keys():
                    dict_piano[piano_composer]['num_file'] += 1
                    dict_piano[piano_composer]['relative_length'] += length_piano
                else:
                    dict_piano[piano_composer] = {}
                    dict_piano[piano_composer]['num_file'] = 1
                    dict_piano[piano_composer]['relative_length'] = length_piano
                T_piano += length_piano

    import pdb; pdb.set_trace()
    for k, v in dict_orch.iteritems():
        dict_orch[k]['relative_length'] = (dict_orch[k]['relative_length'] + 0.) / T_orch
    for k, v in dict_piano.iteritems():
        dict_piano[k]['relative_length'] = (dict_piano[k]['relative_length'] + 0.) / T_piano
    return dict_piano, dict_orch

if __name__ == '__main__':
    get_number_of_file_per_composer([DATABASE_PATH + '/' + e for e in ['bouliane.csv', 'hand_picked_spotify.csv', 'liszt_classical_archives.csv']])