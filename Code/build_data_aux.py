#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import re
import unicodecsv as csv
import numpy as np
from acidano.data_processing.midi.read_midi import get_time, read_midi


def get_instru_and_pr_from_fodler_path(folder_path, quantization):
    # There should be 2 files
    mid_files = glob.glob(folder_path + '/*.mid')
    csv_files = glob.glob(folder_path + '/*.csv')
    # Time
    if len(mid_files) != 2:
        raise Exception('There should be two midi files in ' + folder_path)
    T0 = get_time(mid_files[0], quantization)
    T1 = get_time(mid_files[1], quantization)
    # Instrument and respective pitch
    pr0 = read_midi(mid_files[0], quantization)
    pr1 = read_midi(mid_files[1], quantization)
    if len(csv_files) != 2:
        raise Exception('There should be two csv files in ' + folder_path)
    with open(csv_files[0], 'rb') as f0, open(csv_files[1], 'rb') as f1:
        r0 = csv.DictReader(f0)
        instru0 = next(r0)
        r1 = csv.DictReader(f1)
        instru1 = next(r1)

    return pr0, instru0, T0, pr1, instru1, T1


def instru_pitch_range(instrumentation, pr, instrument_mapping, instrument_list_from_dico):
    for k,v in instrumentation.iteritems():
        pr_instru = pr[k]
        if v in instrument_mapping.keys():
            old_min = instrument_mapping[v]['pitch_min']
            old_max = instrument_mapping[v]['pitch_max']
            # Get the min :
            #   - sum along time dimension
            #   - get the first non-zero index
            this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
            this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0])
            instrument_mapping[v]['pitch_min'] = min(old_min, this_min)
            instrument_mapping[v]['pitch_max'] = max(old_max, this_max)
        else:
            # Sanity check : notations consistency
            if v not in ['double bass', 'english horn']:
                # 'double bass' is the only problematic name.
                # For instance, 'clarinet bass' would match 'clarinet' once suffix is removed
                v_no_suffix = re.split(ur'\s', v)[0]
            else:
                v_no_suffix = v
            if v_no_suffix not in instrument_list_from_dico:
                print 'V PAS DANS INSTRUMENT LISTE'
                import pdb; pdb.set_trace()
            instrument_mapping[v] = {}
            this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
            this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0])
            instrument_mapping[v]['pitch_min'] = this_min
            instrument_mapping[v]['pitch_max'] = this_max
    return instrument_mapping


def cast_small_pr_into_big_pr(pr_small, instru, time, duration, instrument_mapping, pr_big):
    # Detremine x_min and x_max thanks to time and duration
    # Parse pr_small by keys (instrument)
    # Get insrument name in instru
    # For pr_instrument, remove the column out of pitch_min and pitch_max
    # Determine thanks to instrument_mapping the y_min and y_max in pr_big

    # Detremine t_min and t_max thanks to time and duration
    t_min = time
    t_max = time + duration
    # Parse pr_small
    for track_name, pr_instru in pr_small.iteritems():
        if len(instru) == 0:
            # Then this is the piano score
            instru_name = 'piano'
        else:
            instru_name = instru[track_name]
        # For pr_instrument, remove the column out of pitch_min and pitch_max
        pitch_min = instrument_mapping[instru_name]['pitch_min']
        pitch_max = instrument_mapping[instru_name]['pitch_max']

        # Determine thanks to instrument_mapping the y_min and y_max in pr_big
        index_min = instrument_mapping[instru_name]['index_min']
        index_max = instrument_mapping[instru_name]['index_max']

        # Insert the small pr in the big one :)
        pr_big[t_min:t_max, index_min:index_max] = pr_instru[:, pitch_min:pitch_max]

    return pr_big


def warp_pr(pr, T_source, T_target):
    ratio = T_source / float(T_target)
    index_mask = [int(round(x * ratio)) for x in range(0, T_target)]
    pr_return = {}
    for k,v in pr.iteritems():
        pr_return[k] = pr[k][index_mask, :]
    return pr_return
