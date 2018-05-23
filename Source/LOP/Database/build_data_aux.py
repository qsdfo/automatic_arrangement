#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import glob
import re
import csv
from unidecode import unidecode
import numpy as np
from LOP_database.midi.read_midi import Read_midi
from LOP_database.utils.pianoroll_processing import clip_pr, get_pianoroll_time
from LOP_database.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
from LOP_database.utils.event_level import get_event_ind_dict
from LOP_database.utils.pianoroll_processing import sum_along_instru_dim
from LOP_database.utils.align_pianorolls import align_pianorolls
from LOP.Database.simplify_instrumentation import get_simplify_mapping
from LOP.Utils.process_data import process_data_piano, process_data_orch

def get_instru_and_pr_from_folder_path(folder_path, quantization, clip=True):
    # There should be 2 files
    mid_files = glob.glob(folder_path + '/*.mid')
    # Deduce csv files
    csv_files = [re.sub(r'\.mid', '.csv', e) for e in mid_files]

    # Time
    if len(mid_files) != 2:
        raise Exception('There should be two midi files in ' + folder_path)

    def file_processing(path, quantization, clip):
        reader_midi = Read_midi(path, quantization)
        # Read midi
        pianoroll = reader_midi.read_file()
        # Clip
        if clip:
            pianoroll = clip_pr(pianoroll)
        return pianoroll, get_pianoroll_time(pianoroll)

    pianoroll_0, T0 = file_processing(mid_files[0], quantization, clip)
    pianoroll_1, T1 = file_processing(mid_files[1], quantization, clip)

    if len(csv_files) != 2:
        raise Exception('There should be two csv files in ' + folder_path)
    with open(csv_files[0], 'r') as f0, open(csv_files[1], 'r') as f1:
        r0 = csv.DictReader(f0, delimiter=';')
        instru0 = next(r0)
        r1 = csv.DictReader(f1, delimiter=';')
        instru1 = next(r1)

    # Simplify names : keep only tracks not marked as useless
    instru0_simple = {k: simplify_instrumentation(v) for k, v in instru0.items()}
    instru1_simple = {k: simplify_instrumentation(v) for k, v in instru1.items()}
    # Files name, no extensions
    mid_file_0 = re.sub('.mid', '', mid_files[0])
    mid_file_1 = re.sub('.mid', '', mid_files[1])

    return pianoroll_0, instru0_simple, T0, mid_file_0, pianoroll_1, instru1_simple, T1, mid_file_1


def unmixed_instru(instru_string):
    instru_list = re.split(r' and ', instru_string)
    return instru_list


def instru_pitch_range(instrumentation, pr, instru_mapping):
    for k, v in instrumentation.items():
        if k not in pr.keys():
            # BAD BAD BAD
            # Shouldn't happen, but sometimes midi files contain empty tracks
            # listed in the csv file, but not return by the read_midi function...
            # FIX THAT (don't write them in the csv file when generating it)
            continue
        # Get unmixed instru names
        instru_names = unmixed_instru(v)
        # Avoid mixed instrumentation for determining the range.
        # Why ?
        # For instance, tutti strings in the score will be written as a huge chord spanning all violin -> dbass range
        # Hence we don't want range of dbas in violin
        if len(instru_names) > 1:
            continue
        # Corresponding pianoroll
        pr_instru = pr[k]
        if pr_instru.sum() == 0:
            continue
        for instru_name in instru_names:
            # Avoid "Remove" instruments
            if instru_name == 'Remove':
                continue
            if instru_name in instru_mapping.keys():
                old_max = instru_mapping[instru_name]['pitch_max']
                old_min = instru_mapping[instru_name]['pitch_min']
                # Get the min :
                #   - sum along time dimension
                #   - get the first non-zero index
                this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
                this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0]) + 1
                instru_mapping[instru_name]['pitch_min'] = min(old_min, this_min)
                instru_mapping[instru_name]['pitch_max'] = max(old_max, this_max)
            else:
                instru_mapping[instru_name] = {}
                this_min = min(np.nonzero(np.sum(pr_instru, axis=0))[0])
                this_max = max(np.nonzero(np.sum(pr_instru, axis=0))[0]) + 1
                instru_mapping[instru_name]['pitch_min'] = this_min
                instru_mapping[instru_name]['pitch_max'] = this_max
    return instru_mapping


def clean_event(event, trace, trace_prod):
    # Remove from the traces the removed indices
    new_event = []
    counter = 0
    for t, tp in zip(trace, trace_prod):
        if t + tp == 2:
            new_event.append(event[counter])
            counter += 1
        elif t != 0:
            # the t=counter-th event is lost
            counter +=1
    return new_event

def discriminate_between_piano_and_orchestra(pr0, instru0, T0, name0, pr1, instru1, T1, name1):
    if len(set(instru0.values())) > len(set(instru1.values())):
        pr_orch = pr0
        T_orch = T0
        instru_orch = instru0
        name_orch = name0
        #
        pr_piano = pr1
        T_piano = T1
        instru_piano = instru1
        name_piano = name1
    elif len(set(instru0.values())) < len(set(instru1.values())):
        pr_orch = pr1
        T_orch = T1
        instru_orch = instru1
        name_orch = name1
        #
        pr_piano = pr0
        T_piano = T0
        instru_piano = instru0
        name_piano = name0
    else:
        # Both tracks have the same number of instruments
        return [None] * 8
    return pr_piano, instru_piano, T_piano, name_piano, pr_orch, instru_orch, T_orch, name_orch


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
    for track_name, pr_instru in pr_small.items():
        track_name = unidecode(track_name)
        if len(instru) == 0:
            # Then this is the piano score
            instru_names = ['Piano']
        else:
            # unmix instrusi
            track_name_processed = (track_name.rstrip('\x00')).replace('\r', '')
            instru_names = unmixed_instru(instru[track_name_processed])
        
        for instru_name in instru_names:
            # "Remove" tracks
            if instru_name == 'Remove':
                continue

            # For pr_instrument, remove the column out of pitch_min and pitch_max
            try:
                pitch_min = instru_mapping[instru_name]['pitch_min']
                pitch_max = instru_mapping[instru_name]['pitch_max']
            except KeyError:
                print(instru_name + " instrument was not present in the training database")
                continue

            # Determine thanks to instru_mapping the y_min and y_max in pr_big
            index_min = instru_mapping[instru_name]['index_min']
            index_max = instru_mapping[instru_name]['index_max']

            # Insert the small pr in the big one :)
            # Insertion is max between already written notes and new ones
            pr_big[t_min:t_max, index_min:index_max] = np.maximum(pr_big[t_min:t_max, index_min:index_max], pr_instru[:, pitch_min:pitch_max])
            
    return pr_big

def simplify_instrumentation(instru_name_complex):
    simplify_mapping = get_simplify_mapping()
    instru_name_unmixed = unmixed_instru(instru_name_complex)
    instru_name_unmixed_simple = []
    for e in instru_name_unmixed:
        simple_name = simplify_mapping[e]
        instru_name_unmixed_simple.append(simple_name)
    link = " and "
    return link.join(instru_name_unmixed_simple)

def process_folder(folder_path, quantization, binary_piano, binary_orch, temporal_granularity, gapopen=3, gapextend=1):
    # Get instrus and prs from a folder name name
    pr0, instru0, T0, name0, pr1, instru1, T1, name1 = get_instru_and_pr_from_folder_path(folder_path, quantization)

    pr_piano, instru_piano, T_piano, name_piano, pr_orch, instru_orch, T_orch, name_orch=\
            discriminate_between_piano_and_orchestra(pr0, instru0, T0, name0, pr1, instru1, T1, name1)

    pr_piano = process_data_piano(pr_piano, binary_piano)
    pr_orch = process_data_orch(pr_orch, binary_orch)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        event_piano = get_event_ind_dict(pr_piano)
        event_orch = get_event_ind_dict(pr_orch)
        def get_duration(event, last_time):
            start_ind = event[:]
            end_ind = np.zeros(event.shape, dtype=np.int)
            end_ind[:-1] = event[1:]
            end_ind[-1] = last_time
            duration_list = end_ind - start_ind
            return duration_list
        duration_piano = get_duration(event_piano, T_piano)
        duration_orch = get_duration(event_orch, T_orch)
        # Get the duration of each event
        pr_piano = warp_pr_aux(pr_piano, event_piano)
        pr_orch = warp_pr_aux(pr_orch, event_orch)
    else:
        event_piano = None
        event_orch = None

    # Align tracks
    piano_aligned, trace_piano, orch_aligned, trace_orch, trace_prod, total_time = align_pianorolls(pr_piano, pr_orch, gapopen, gapextend)
    
    # Clean events
    if (temporal_granularity == 'event_level'):
        if (trace_piano is None) or (trace_orch is None):
            event_piano_aligned = None
            event_orch_aligned = None
            duration_piano_aligned = None
            duration_orch_aligned = None
        else:
            event_piano_aligned = clean_event(event_piano, trace_piano, trace_prod)
            event_orch_aligned = clean_event(event_orch, trace_orch, trace_prod)
            duration_piano_aligned = clean_event(duration_piano, trace_piano, trace_prod)
            duration_orch_aligned = clean_event(duration_orch, trace_orch, trace_prod)
    else:
        event_piano_aligned = []
        event_orch_aligned = []
        duration_piano_aligned = []
        duration_orch_aligned = []

    return piano_aligned, event_piano, duration_piano, instru_piano, name_piano, orch_aligned, event_orch, duration_orch, instru_orch, name_orch, total_time

if __name__ == '__main__':
    pr_piano, event_piano, duration_piano, instru_piano, name_piano, pr_orch, event_orch, duration_orch, instru_orch, name_orch, total_time = process_folder('/Users/leo/Recherche/automatic_orchestration/database/Orchestration/LOP_database_06_09_17/bouliane/0', 8, True, True, 'event_level')

    

