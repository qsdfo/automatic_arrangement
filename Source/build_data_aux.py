#!/usr/bin/env python
# -*- coding: utf8 -*-

import glob
import re
import unicodecsv as csv
from unidecode import unidecode
import numpy as np
from acidano.data_processing.midi.read_midi import Read_midi
from acidano.data_processing.utils.pianoroll_processing import clip_pr, get_pianoroll_time
from acidano.data_processing.utils.time_warping import needleman_chord_wrapper, warp_dictionnary_trace, remove_zero_in_trace, warp_pr_aux
from acidano.data_processing.utils.event_level import get_event_ind_dict
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
import acidano.data_processing.utils.unit_type as Unit_type

def get_instru_and_pr_from_folder_path(folder_path, quantization, clip=True):
    # There should be 2 files
    mid_files = glob.glob(folder_path + '/*.mid')
    # Deduce csv files
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

    # Simplify names
    instru0_simple = {k:simplify_instrumentation(v) for k,v in instru0.iteritems()}
    instru1_simple = {k:simplify_instrumentation(v) for k,v in instru1.iteritems()}

    # Files name, no extensions
    mid_file_0 = re.sub('\.mid', '', mid_files[0])
    mid_file_1 = re.sub('\.mid', '', mid_files[1])

    return pianoroll_0, instru0_simple, T0, mid_file_0, pianoroll_1, instru1_simple, T1, mid_file_1


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
        # Get unmixed instru names
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
            if instru_name in instru_mapping.keys():
                ### ???
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


def align_tracks(pr0, pr1, unit_type, gapopen, gapextend):
    # Get trace from needleman_wunsch algorithm
    # Traces are computed from binaries matrices
    # Traces are binary lists, 0 meaning a gap is inserted
    pr0_trace = sum_along_instru_dim(Unit_type.from_type_to_binary(pr0, unit_type))
    pr1_trace = sum_along_instru_dim(Unit_type.from_type_to_binary(pr1, unit_type))
    trace_0, trace_1, this_sum_score, this_nbId, this_nbDiffs = needleman_chord_wrapper(pr0_trace, pr1_trace, gapopen, gapextend)

    # Wrap dictionnaries according to the traces
    assert(len(trace_0) == len(trace_1)), "size mismatch"
    pr0_warp = warp_dictionnary_trace(pr0, trace_0)
    pr1_warp = warp_dictionnary_trace(pr1, trace_1)
    # Get pr warped and duration# In fact we just discard 0 in traces for both pr

    trace_prod = [e1 * e2 for (e1,e2) in zip(trace_0, trace_1)]
    duration = sum(trace_prod)
    if duration == 0:
        return [None]*2
    pr0_aligned = remove_zero_in_trace(pr0_warp, trace_prod)
    pr1_aligned = remove_zero_in_trace(pr1_warp, trace_prod)

    return pr0_aligned, trace_0, pr1_aligned, trace_1, trace_prod, duration

def discriminate_between_piano_and_orchestra(pr0, instru0, name0, pr1, instru1, name1):
    if len(set(instru0.values())) > len(set(instru1.values())):
        pr_orchestra = pr0
        instru_orchestra = instru0
        name_orchestra = name0
        pr_piano = pr1
        instru_piano = instru1
        name_piano = name1
    elif len(set(instru0.values())) < len(set(instru1.values())):
        pr_orchestra = pr1
        instru_orchestra = instru1
        name_orchestra = name1
        pr_piano = pr0
        instru_piano = instru0
        name_piano = name0
    else:
        logging.info('The two midi files have the same number of instruments')
    return pr_piano, instru_piano, name_piano, pr_orchestra, instru_orchestra, name_orchestra

def process_folder(folder_path, quantization, unit_type, temporal_granularity, logging, gapopen=3, gapextend=1):
    # Get instrus and prs from a folder name name
    pr0, instru0, _, name0, pr1, instru1, _, name1 = get_instru_and_pr_from_folder_path(folder_path, quantization)

    # Unit type
    pr0 = Unit_type.from_rawpr_to_type(pr0, unit_type)
    pr1 = Unit_type.from_rawpr_to_type(pr1, unit_type)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        pr0 = warp_pr_aux(pr0, get_event_ind_dict(pr0))
        pr1 = warp_pr_aux(pr1, get_event_ind_dict(pr1))

    # Align tracks
    pr0_aligned, _, pr1_aligned, _, _, duration = align_tracks(pr0, pr1, unit_type, gapopen, gapextend)

    # Find which pr is orchestra, which one is piano
    pr_piano, instru_piano, name_piano, pr_orchestra, instru_orchestra, name_orchestra =\
        discriminate_between_piano_and_orchestra(pr0_aligned, instru0, name0, pr1_aligned, instru1, name1)

    return pr_piano, instru_piano, name_piano, pr_orchestra, instru_orchestra, name_orchestra, duration


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
            instru_names = unmixed_instru(instru[track_name])

        for instru_name in instru_names:
            # For pr_instrument, remove the column out of pitch_min and pitch_max
            pitch_min = instru_mapping[instru_name]['pitch_min']
            pitch_max = instru_mapping[instru_name]['pitch_max']

            # Determine thanks to instru_mapping the y_min and y_max in pr_big
            index_min = instru_mapping[instru_name]['index_min']
            index_max = instru_mapping[instru_name]['index_max']

            # Insert the small pr in the big one :)
            # Insertion is max between already written notes and new ones
            pr_big[t_min:t_max, index_min:index_max] = np.maximum(pr_big[t_min:t_max, index_min:index_max], pr_instru[:, pitch_min:pitch_max])

    return pr_big


def simplify_instrumentation(instru_name_complex):
    # In the Orchestration_checked folder, midi files are associated to csv files
    # This script aims at reducing the number of instrument written in the csv files
    # by grouping together different but close instruments ()
    simplify_mapping = {
        # u'tuba bass': u'tuba',
        u'voice soprano mezzo': u'voice',
        u'bell': u'percussion',
        u'voice baritone': u'voice',
        # 'piccolo': 'piccolo',
        u'trombone tenor': u'trombone',
        # 'celesta': 'celesta',
        # 'horn': 'horn',
        u'castanets': u'percussion',
        # 'euphonium': 'euphonium',
        # 'lyre': 'lyre',
        # 'english horn': 'english horn',
        # 'trombone': 'trombone',
        # 'violin': 'violin',
        # 'clarinet': 'clarinet',
        # 'trumpet': 'trumpet',
        # u'cornet': u'trumpet',
        # 'bassoon': 'bassoon',
        u'trombone bass': u'trombone',
        # 'timpani': 'timpani',
        # 'tuba': 'tuba',
        # 'percussion': 'percussion',
        # 'violoncello': 'violoncello',
        # u'bassoon bass': u'bassoon',
        # 'viola': 'viola',
        # 'piano': 'piano',
        # 'harp': 'harp',
        u'voice soprano': u'voice',
        u'triangle': u'percussion',
        u'trombone alto tenor': u'trombone',
        # 'oboe': 'oboe',
        u'drum bass': u'percussion',
        # 'flute':,
        u'cymbal': u'percussion',
        u'trombone alto': u'trombone',
        u'glockenspiel': u'percussion',
        u'voice alto': u'voice',
        u'tam tam': u'percussion',
        u'drum': u'percussion',
        # 'organ':,
        u'voice bass': u'voice',
        # u'clarinet bass': u'clarinet',
        # 'double bass':,
        # 'saxophone':,
        # 'voice':,
        u'voice tenor': u'voice',
        u'harpsichord': u'piano'  # A bit scandalous, but actually never present alone in the corpus, only as a mixed instrument, which causes problems
    }
    instru_name_unmixed = unmixed_instru(instru_name_complex)
    instru_name_unmixed_simple = []
    for e in instru_name_unmixed:
        if e in simplify_mapping:
            instru_name_unmixed_simple.append(simplify_mapping[e])
        else:
            instru_name_unmixed_simple.append(e)
    link = " and "
    return link.join(instru_name_unmixed_simple)

if __name__=='__main__':
    name = 'DEBUG/test.mid'
    reader = Read_midi(name, 12)
    time = reader.get_time()
    pr = reader.read_file()
