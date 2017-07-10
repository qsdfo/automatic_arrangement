import csv
import build_data_aux
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATABASE_PATH = "/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_30_06_17"
DATA_PATH = "../../Data"

def get_number_of_file_per_composer(files):
    
    dict_orch = {}
    dict_piano = {}
    T_orch = 0
    T_piano = 0 

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
                pianoroll_0, instru_0, _, _, pianoroll_1, instru_1, _, _ = build_data_aux.get_instru_and_pr_from_folder_path(path, 100)
                pr_piano, _, _, _,\
                pr_orch, _, _, _,\
                _ =\
                build_data_aux.discriminate_between_piano_and_orchestra(pianoroll_0, 0, instru_0, 0,
                                                         pianoroll_1, 0, instru_1, 0,
                                                         0)

                if (pr_piano == None) or (pr_orch == None):
                    print("Skipping " + path)
                    continue 

                try:
                    length_piano = len(sum_along_instru_dim(pr_piano))
                    length_orch = len(sum_along_instru_dim(pr_orch))
                except:
                    import pdb; pdb.set_trace()

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
    
    for k, v in dict_orch.iteritems():
        dict_orch[k]['relative_length'] = (100 * dict_orch[k]['relative_length'] + 0.) / T_orch
    for k, v in dict_piano.iteritems():
        dict_piano[k]['relative_length'] = (100 * dict_piano[k]['relative_length'] + 0.) / T_piano

    # Merge the two dictionnaries
    merge_dict = {}
    for k, v in dict_piano.iteritems():
        merge_dict[k]  = {'relative_length_piano': v['relative_length'],
                     'relative_length_orch': None,
                     'num_file_piano': v['num_file'],
                     'num_file_orch': None,
                     }

    for k, v in dict_orch.iteritems():
        if k in merge_dict.keys():
            merge_dict[k]['relative_length_orch'] = v['relative_length']
            merge_dict[k]['num_file_orch'] = v['num_file']
        else:   
            merge_dict[k]  = {'relative_length_piano': None,
                         'relative_length_orch': v['relative_length'],
                         'num_file_piano': None,
                         'num_file_orch': v['num_file'],
                         }

    # Replace '' by unknown
    merge_dict['unknown'] = merge_dict.pop('', None)
    merge_dict_name = {}
    for k, v in merge_dict.iteritems():
        name_list = re.split(' ', k)
        new_name = name_list[-1] + '. ' + ' '.join(name_list[:-1])
        merge_dict_name[new_name] = v
    merge_dict = merge_dict_name
    
    # write a csv file containing these information
    def float_to_string(s):
        return '' if s is None else "{:2.2f}".format(s)
    def format_none(s):
        return '' if s is None else "{}".format(s)
    with open('stat_composer.csv', 'wb') as f:
        f.write("composer,numpiano,percentagepiano,numorch,percentageorch\n")
        for k, v in sorted(merge_dict.items()):
            # f.write("{};{};{:2.2f};{};{:2.2f}".format(k, v['num_file_piano'], v['relative_length_piano'], v['num_file_orch'], v['relative_length_orch']))
            f.write("{},{},{},{},{}\n".format(format_none(k).title(), format_none(v['num_file_piano']), float_to_string(v['relative_length_piano']), format_none(v['num_file_orch']), float_to_string(v['relative_length_orch'])))
    
    # barplot
    fig, ax = plt.subplots()
    width = 0.35
    x = []
    y = []
    for k, v in merge_dict.iteritems():
        if v['relative_length_orch']:
            x.append(k[:3])
            y.append(v['relative_length_orch'])
    ind = range(len(y)) 
    barplot = ax.bar(ind, y, width, alpha=0.7, color='b')

    ax.set_title('Representativeness of composers in the database', fontsize=14, fontweight='bold')
    ax.set_xlabel('Composers')
    ax.set_ylabel('Ratio of frame occurences per composer')
    ax.set_xticks([e + width / 2 for e in ind])
    ax.set_xticklabels(x)
    plt.savefig('ratio_composer.pdf')

    return

def histogram_per_pitch():
    # Bar plot of the occurences for each pitch
    # Load orch matrices
    orchestra_train = np.load(DATA_PATH + '/orchestra_train.npy')
    orchestra_test = np.load(DATA_PATH + '/orchestra_test.npy')
    orchestra_valid = np.load(DATA_PATH + '/orchestra_valid.npy')
    N_orch = orchestra_train.shape[1]
    T_orch = orchestra_train.shape[0] + orchestra_test.shape[0] + orchestra_valid.shape[0]
    orch_on = (orchestra_train.sum(axis=0) + orchestra_test.sum(axis=0) + orchestra_valid.sum(axis=0) + 0.) / T_orch

    # Plot barplot
    bar_width = 0.35
    x = range(len(orch_on))
    barplot = plt.bar(x, orch_on, bar_width,
                 alpha=0.7,
                 color='b',
                 label='Number notes on')


    plt.suptitle('Activation ratio per pitch in the whole database', fontsize=14, fontweight='bold')
    plt.xlabel('pitch')
    plt.ylabel('nb_occurence / total_length')
    plt.savefig('barplot_pitch_interclass_imbalance.pdf')


def label_cardinality():
    orchestra_train = np.load(DATA_PATH + '/orchestra_train.npy')
    orchestra_test = np.load(DATA_PATH + '/orchestra_test.npy')
    orchestra_valid = np.load(DATA_PATH + '/orchestra_valid.npy')
    orchestra = np.concatenate((orchestra_train, orchestra_valid, orchestra_test), axis=0)
    import pdb; pdb.set_trace()

    return np.mean(orchestra.sum(axis=1))

if __name__ == '__main__':
    # get_number_of_file_per_composer([DATABASE_PATH + '/' + e for e in ['bouliane.csv', 'hand_picked_spotify.csv', 'liszt_classical_archives.csv', 'imslp.csv']])

    # histogram_per_pitch()

    print(label_cardinality())
    