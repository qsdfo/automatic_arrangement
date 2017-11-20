#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

import glob
import re
import unicodecsv as csv
from unidecode import unidecode
import numpy as np
from LOP_database.midi.read_midi import Read_midi
from LOP_database.utils.pianoroll_processing import clip_pr, get_pianoroll_time
from LOP_database.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
from LOP_database.utils.event_level import get_event_ind_dict
from LOP_database.utils.pianoroll_processing import sum_along_instru_dim
from LOP_database.utils.pianoroll_reduction import remove_unmatched_silence, remove_match_silence, remove_silence
from LOP_database.utils.align_pianorolls import align_pianorolls
from simplify_instrumentation import get_simplify_mapping

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

    # Simplify names : keep only tracks not marked as useless
    instru0_simple = {k: simplify_instrumentation(v) for k, v in instru0.iteritems()}
    instru1_simple = {k: simplify_instrumentation(v) for k, v in instru1.iteritems()}
    # Files name, no extensions
    mid_file_0 = re.sub('\.mid', '', mid_files[0])
    mid_file_1 = re.sub('\.mid', '', mid_files[1])

    return pianoroll_0, instru0_simple, T0, mid_file_0, pianoroll_1, instru1_simple, T1, mid_file_1


def unmixed_instru(instru_string):
    instru_list = re.split(ur' and ', instru_string)
    return instru_list


def instru_pitch_range(instrumentation, pr, instru_mapping, instrument_list_from_dico):
    for k, v in instrumentation.iteritems():
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
                # ???
                # v_no_suffix = re.split(ur'\s', instru_name)[0]
                #######
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


def discriminate_between_piano_and_orchestra(pr0, event0, duration0, instru0, name0, pr1, event1, duration1, instru1, name1, duration):
    if len(set(instru0.values())) > len(set(instru1.values())):
        pr_orch = pr0
        event_orch = event0
        duration_orch = duration0
        instru_orch = instru0
        name_orch = name0
        #
        pr_piano = pr1
        event_piano = event1
        duration_piano = duration1
        instru_piano = instru1
        name_piano = name1
    elif len(set(instru0.values())) < len(set(instru1.values())):
        pr_orch = pr1
        event_orch = event1
        duration_orch = duration1
        instru_orch = instru1
        name_orch = name1
        #
        pr_piano = pr0
        event_piano = event0
        duration_piano = duration0
        instru_piano = instru0
        name_piano = name0
    else:
        # Both tracks have the same number of instruments
        return [None] * 11
    return pr_piano, event_piano, duration_piano, instru_piano, name_piano, pr_orch, event_orch, duration_orch, instru_orch, name_orch, duration


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
            instru_names = ['Piano']
        else:
            # unmix instrusi
            instru_names = unmixed_instru(instru[track_name.rstrip('\x00')])
        
        for instru_name in instru_names:
            # "Remove" tracks
            if instru_name == 'Remove':
                continue

            # For pr_instrument, remove the column out of pitch_min and pitch_max
            pitch_min = instru_mapping[instru_name]['pitch_min']
            pitch_max = instru_mapping[instru_name]['pitch_max']

            # Determine thanks to instru_mapping the y_min and y_max in pr_big
            index_min = instru_mapping[instru_name]['index_min']
            index_max = instru_mapping[instru_name]['index_max']

            # Insert the small pr in the big one :)
            # Insertion is max between already written notes and new ones
            try:
                pr_big[t_min:t_max, index_min:index_max] = np.maximum(pr_big[t_min:t_max, index_min:index_max], pr_instru[:, pitch_min:pitch_max])
            except:
                import pdb; pdb.set_trace()

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

def process_folder(folder_path, quantization, temporal_granularity, gapopen=3, gapextend=1):
    # Get instrus and prs from a folder name name
    pr0, instru0, T0, name0, pr1, instru1, T1, name1 = get_instru_and_pr_from_folder_path(folder_path, quantization)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        event_0 = get_event_ind_dict(pr0)
        event_1 = get_event_ind_dict(pr1)
        def get_duration(event, last_time):
            start_ind = event[:]
            end_ind = np.zeros(event.shape, dtype=np.int)
            end_ind[:-1] = event[1:]
            end_ind[-1] = last_time
            duration_list = end_ind - start_ind
            return duration_list
        duration0 = get_duration(event_0, T0)
        duration1 = get_duration(event_1, T1)
        # Get the duration of each event
        pr0 = warp_pr_aux(pr0, event_0)
        pr1 = warp_pr_aux(pr1, event_1)
    else:
        event_0 = None
        event_1 = None

    # Align tracks
    pr0_aligned, trace_0, pr1_aligned, trace_1, trace_prod, duration = align_pianorolls(pr0, pr1, gapopen, gapextend)
    
    # Clean events
    if (temporal_granularity == 'event_level'):
        if (trace_0 is None) or (trace_1 is None):
            event0_aligned = None
            event1_aligned = None
            duration0_aligned = None
            duration1_aligned = None
        else:
            event0_aligned = clean_event(event_0, trace_0, trace_prod)
            event1_aligned = clean_event(event_1, trace_1, trace_prod)
            duration0_aligned = clean_event(duration0, trace_0, trace_prod)
            duration1_aligned = clean_event(duration1, trace_1, trace_prod)
    else:
        event0_aligned = []
        event1_aligned = []
        duration0_aligned = []
        duration1_aligned = []

    # Find which pr is orchestra, which one is piano
    pr_piano, event_piano, duration_piano, instru_piano, name_piano,\
        pr_orch, event_orch, duration_orch, instru_orch, name_orch,\
        duration =\
            discriminate_between_piano_and_orchestra(pr0_aligned, event0_aligned, duration0_aligned, instru0, name0,
                                                 pr1_aligned, event1_aligned, duration1_aligned, instru1, name1,
                                                 duration)

    return pr_piano, event_piano, duration_piano, instru_piano, name_piano, pr_orch, event_orch, duration_orch, instru_orch, name_orch, duration


if __name__ == '__main__':
    process_folder('/home/aciditeam-leo/Aciditeam/database/Orchestration/LOP_database_30_06_17/imslp/72', 4, 'binary', 'frame_level')
