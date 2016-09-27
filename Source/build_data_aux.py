#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import re
import unicodecsv as csv
from unidecode import unidecode
import numpy as np
from acidano.data_processing.midi.read_midi import Read_midi
from acidano.data_processing.utils.pianoroll_processing import clip_pr, get_pianoroll_time
from acidano.data_processing.midi.write_midi import write_midi


def get_instru_and_pr_from_folder_path(folder_path, quantization, clip=True):
    # There should be 2 files
    mid_files = glob.glob(folder_path + '/*.mid')
    csv_files = glob.glob(folder_path + '/*.csv')
    # Time
    if len(mid_files) != 2:
        raise Exception('There should be two midi files in ' + folder_path)

    def file_processing(path, quantization, clip):
        reader_midi = Read_midi(path, quantization)
        # Read midi
        reader_midi.read_file()
        pianoroll = reader_midi.pianoroll
        # Clip
        if clip:
            pianoroll = clip_pr(pianoroll)
        return pianoroll, get_pianoroll_time(pianoroll)

    pianoroll_0, T0 = file_processing(mid_files[0], quantization, clip)
    pianoroll_1, T1 = file_processing(mid_files[1], quantization, clip)

    if len(csv_files) != 2:
        raise Exception('There should be two csv files in ' + folder_path)
    with open(csv_files[0], 'rb') as f0, open(csv_files[1], 'rb') as f1:
        r0 = csv.DictReader(f0, delimiter=';')
        instru0 = next(r0)
        r1 = csv.DictReader(f1, delimiter=';')
        instru1 = next(r1)

    return pianoroll_0, instru0, T0, mid_files[0], pianoroll_1, instru1, T1, mid_files[1]


def unmixed_instru(instru_string):
    instru_list = re.split(ur' and ', instru_string)
    return instru_list


def instru_pitch_range(instrumentation, pr, instru_mapping, instrument_list_from_dico):
    for k,v in instrumentation.iteritems():
        if k not in pr.keys():
            # BAD BAD BAD
            # Shouldn't happen, but sometimes midi files contain empty tracks
            # listed in the csv file, but not return by the read_midi function...
            # FIX THAT (don't write them in the csv file when generating it)
            continue
        instru_liste = unmixed_instru(v)
        if len(instru_liste) > 1:
            # As explained in the readme of build_data.py function,
            # instrumental mixes are not taken into consideration for pitch ranges
            continue
        pr_instru = pr[k]
        if v in instru_mapping.keys():
            old_min = instru_mapping[v]['pitch_min']
            old_max = instru_mapping[v]['pitch_max']
            # Get the min :
            #   - sum along time dimension
            #   - get the first non-zero index
            this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
            this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0])
            instru_mapping[v]['pitch_min'] = min(old_min, this_min)
            instru_mapping[v]['pitch_max'] = max(old_max, this_max)
        else:
            # Sanity check : notations consistency
            v_no_suffix = re.split(ur'\s', v)[0]
            if (v_no_suffix not in instrument_list_from_dico) and (v not in instrument_list_from_dico):
                print 'V PAS DANS INSTRUMENT LISTE'
                import pdb; pdb.set_trace()
            instru_mapping[v] = {}
            this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
            this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0])
            instru_mapping[v]['pitch_min'] = this_min
            instru_mapping[v]['pitch_max'] = this_max
    return instru_mapping


def cast_small_pr_into_big_pr(pr_small, instru, time, duration, instru_mapping, pr_big):
    # Detremine x_min and x_max thanks to time and duration
    # Parse pr_small by keys (instrument)
    # Get insrument name in instru
    # For pr_instrument, remove the column out of pitch_min and pitch_max
    # Determine thanks to instru_mapping the y_min and y_max in pr_big

    # Detremine t_min and t_max thanks to time and duration
    t_min = time
    t_max = time + duration
    # Parse pr_small
    for track_name, pr_instru in pr_small.iteritems():
        track_name = unidecode(track_name)
        if len(instru) == 0:
            # Then this is the piano score
            instru_names = ['piano']
        else:
            # unmix instrus
            # The pitch selection (pitch_min and pitch_max) will automatically remove
            # pitch outliers
            try:
                instru_names = unmixed_instru(instru[track_name])
            except:
                print track_name + ' not in instru'
                import pdb; pdb.set_trace()

        for instru_name in instru_names:
            # Little hack for a little problem i ran into :
            # harpsichord was only present as a mixed instrument in the database
            # Then it never appears in the instrument mapping though
            # it is a track name for valid names...
            if instru_name not in instru_mapping.keys():
                print instru_name + ' does not have pitch and indices ranges'
                continue
            # For pr_instrument, remove the column out of pitch_min and pitch_max
            pitch_min = instru_mapping[instru_name]['pitch_min']
            pitch_max = instru_mapping[instru_name]['pitch_max']

            # Determine thanks to instru_mapping the y_min and y_max in pr_big
            index_min = instru_mapping[instru_name]['index_min']
            index_max = instru_mapping[instru_name]['index_max']

            # Insert the small pr in the big one :)
            try:
                pr_big[t_min:t_max, index_min:index_max] = pr_instru[:, pitch_min:pitch_max]
            except:
                import pdb; pdb.set_trace()

    return pr_big


if __name__=='__main__':
    name = 'DEBUG/test.mid'
    reader = Read_midi(name, 12)
    time = reader.get_time()
    pr = reader.read_file()
