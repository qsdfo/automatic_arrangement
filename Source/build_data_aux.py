#!/usr/bin/env python
# -*- coding: utf-8-unix -*-

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
from acidano.data_processing.utils.pianoroll_reduction import remove_unmatched_silence, remove_match_silence, remove_silence
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


def align_tracks_removing_silences(pr0, pr1, unit_type, gapopen, gapextend):
    # Mapping_0 is a vector of size pr0, with values of the index in the new pr and -1 for a silence
    # Remove zeros
    pr0_no_silence, mapping_0 = remove_silence(pr0)
    pr1_no_silence, mapping_1 = remove_silence(pr1)

    # Get trace from needleman_wunsch algorithm
    # Traces are computed from binaries matrices
    # Traces are binary lists, 0 meaning a gap is inserted
    pr0_trace = sum_along_instru_dim(Unit_type.from_type_to_binary(pr0_no_silence, unit_type))
    pr1_trace = sum_along_instru_dim(Unit_type.from_type_to_binary(pr1_no_silence, unit_type))
    trace_0, trace_1, this_sum_score, this_nbId, this_nbDiffs = needleman_chord_wrapper(pr0_trace, pr1_trace, gapopen, gapextend)

    ####################################
    ####################################
    # Wrap dictionnaries according to the traces
    assert(len(trace_0) == len(trace_1)), "size mismatch"
    pr0_warp = warp_dictionnary_trace(pr0_no_silence, trace_0)
    pr1_warp = warp_dictionnary_trace(pr1_no_silence, trace_1)
    ####################################
    ####################################

    ####################################
    ####################################
    # Trace product
    trace_prod = [e1 * e2 for (e1, e2) in zip(trace_0, trace_1)]
    duration = sum(trace_prod)
    if duration == 0:
        return [None]*5
    # Remove gaps
    pr0_aligned = remove_zero_in_trace(pr0_warp, trace_prod)
    pr1_aligned = remove_zero_in_trace(pr1_warp, trace_prod)

    # New mapping :
    # some element are lost, replace them by silences
    # and decrease the indices
    def remove_gaps_mapping(mapping, trace, trace_prod):
        index = 0
        counter = 0
        for (t, t_prod) in zip(trace, trace_prod):
            if (t == 1) and (t_prod == 0):
                # Element lost
                while(mapping[index] == -1):
                    index += 1
                # Replace i with a silence
                mapping[index] = -1
            elif (t == 1) and (t_prod == 1):
                while(mapping[index] == -1):
                    index += 1
                mapping[index] = counter
                counter += 1
                index += 1
        return mapping
    mapping_0_aligned = remove_gaps_mapping(mapping_0, trace_0, trace_prod)
    mapping_1_aligned = remove_gaps_mapping(mapping_1, trace_1, trace_prod)
    # Actually it is easier to have the indices of the non silent frames in the original score :
    non_silent_0 = np.where(mapping_0_aligned != -1)[0]
    non_silent_1 = np.where(mapping_1_aligned != -1)[0]
    ####################################
    ####################################

    # # Remove zeros in one score, but not in the other
    # pr0_no_unmatched_silence, pr1_no_unmatched_silence, duration = remove_unmatched_silence(pr0_aligned, pr1_aligned)
    #
    # # Remove zeros in both piano and orchestra : we don't want to learn mapping from zero to zero
    # pr0_out, pr1_out, duration = remove_match_silence(pr0_no_unmatched_silence, pr1_no_unmatched_silence)

    return pr0_aligned, non_silent_0, pr1_aligned, non_silent_1, duration


def align_tracks(pr0, pr1, unit_type, gapopen, gapextend):
    # Get trace from needleman_wunsch algorithm

    # First extract binary representation, whatever unit_type is
    pr0_trace = sum_along_instru_dim(Unit_type.from_type_to_binary(pr0, unit_type))
    pr1_trace = sum_along_instru_dim(Unit_type.from_type_to_binary(pr1, unit_type))

    # Traces are computed from binaries matrices
    # Traces are binary lists, 0 meaning a gap is inserted
    trace_0, trace_1, this_sum_score, this_nbId, this_nbDiffs = needleman_chord_wrapper(pr0_trace, pr1_trace, gapopen, gapextend)

    ####################################
    # Wrap dictionnaries according to the traces
    assert(len(trace_0) == len(trace_1)), "size mismatch"
    pr0_warp = warp_dictionnary_trace(pr0, trace_0)
    pr1_warp = warp_dictionnary_trace(pr1, trace_1)

    ####################################
    # Trace product
    trace_prod = [e1 * e2 for (e1, e2) in zip(trace_0, trace_1)]
    duration = sum(trace_prod)
    if duration == 0:
        return [None]*5
    # Remove gaps
    pr0_aligned = remove_zero_in_trace(pr0_warp, trace_prod)
    pr1_aligned = remove_zero_in_trace(pr1_warp, trace_prod)

    return pr0_aligned, trace_0, pr1_aligned, trace_1, trace_prod, duration


def clean_event(event, trace, trace_prod):
    new_event = []
    counter = 0
    for t, tp in zip(trace, trace_prod):
        if t + tp == 2:
            new_event.append(event[counter])
            counter += 1
        elif t != 0:
            counter +=1
    return new_event


def discriminate_between_piano_and_orchestra(pr0, event0, instru0, name0, pr1, event1, instru1, name1, duration):
    if len(set(instru0.values())) > len(set(instru1.values())):
        pr_orch = pr0
        event_orch = event0
        instru_orch = instru0
        name_orch = name0
        #
        pr_piano = pr1
        event_piano = event1
        instru_piano = instru1
        name_piano = name1
    elif len(set(instru0.values())) < len(set(instru1.values())):
        pr_orch = pr1
        event_orch = event1
        instru_orch = instru1
        name_orch = name1
        #
        pr_piano = pr0
        event_piano = event0
        instru_piano = instru0
        name_piano = name0
    else:
        # Both tracks have the same number of instruments
        return [None] * 11
    return pr_piano, event_piano, instru_piano, name_piano, pr_orch, event_orch, instru_orch, name_orch, duration


def process_folder(folder_path, quantization, unit_type, temporal_granularity, gapopen=3, gapextend=1):
    # Get instrus and prs from a folder name name
    pr0, instru0, _, name0, pr1, instru1, _, name1 = get_instru_and_pr_from_folder_path(folder_path, quantization)

    # Unit type
    pr0 = Unit_type.from_rawpr_to_type(pr0, unit_type)
    pr1 = Unit_type.from_rawpr_to_type(pr1, unit_type)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        event_0 = get_event_ind_dict(pr0)
        event_1 = get_event_ind_dict(pr1)
        pr0 = warp_pr_aux(pr0, event_0)
        pr1 = warp_pr_aux(pr1, event_1)
    else:
        event_0 = None
        event_1 = None

    # Align tracks
    pr0_aligned, trace_0, pr1_aligned, trace_1, trace_prod, duration = align_tracks(pr0, pr1, unit_type, gapopen, gapextend)

    # Clean events
    event0_aligned = clean_event(event_0, trace_0, trace_prod)
    event1_aligned = clean_event(event_1, trace_1, trace_prod)

    # Find which pr is orchestra, which one is piano
    pr_piano, event_piano, instru_piano, name_piano,\
        pr_orch, event_orch, instru_orch, name_orch,\
        duration =\
        discriminate_between_piano_and_orchestra(pr0_aligned, event0_aligned, instru0, name0,
                                                 pr1_aligned, event1_aligned, instru1, name1,
                                                 duration)

    return pr_piano, event_piano, instru_piano, name_piano, pr_orch, event_orch, instru_orch, name_orch, duration


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
            # "Remove" tracks
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
            pr_big[t_min:t_max, index_min:index_max] = np.maximum(pr_big[t_min:t_max, index_min:index_max], pr_instru[:, pitch_min:pitch_max])

    return pr_big


def simplify_instrumentation(instru_name_complex):
    # In the Orchestration_checked folder, midi files are associated to csv files
    # This script aims at reducing the number of instrument written in the csv files
    # by grouping together different but close instruments ()

    # simplify_mapping = {
    #     "Piccolo": "Piccolo",
    #     "Flute": "Flute",
    #     "Alto-Flute": "Flute",
    #     "Soprano-Flute": "Flute",
    #     "Bass-Flute": "Flute",
    #     "Contrabass-Flute": "Flute",
    #     "Pan Flute": "Flute",
    #     "Recorder": "Flute",
    #     "Ocarina": "Remove",
    #     "Oboe": "Oboe",
    #     "Oboe-dAmore": "Oboe",
    #     "Oboe de Caccia": "Oboe",
    #     "English-Horn": "Horn",
    #     "Heckelphone": "Remove",
    #     "Piccolo-Clarinet-Ab": "Clarinet",
    #     "Clarinet": "Clarinet",
    #     "Clarinet-Eb": "Clarinet",
    #     "Clarinet-Bb": "Clarinet",
    #     "Piccolo-Clarinet-D": "Clarinet",
    #     "Clarinet-C": "Clarinet",
    #     "Clarinet-A": "Clarinet",
    #     "Basset-Horn-F": "Horn",
    #     "Alto-Clarinet-Eb": "Clarinet",
    #     "Bass-Clarinet-Bb": "Clarinet",
    #     "Bass-Clarinet-A": "Clarinet",
    #     "Contra-Alto-Clarinet-Eb": "Clarinet",
    #     "Contrabass-Clarinet-Bb": "Clarinet",
    #     "Bassoon": "Bassoon",
    #     "Contrabassoon": "Bassoon",
    #     "Soprano-Sax": "Sax",
    #     "Alto-Sax": "Sax",
    #     "Tenor-Sax": "Sax",
    #     "Baritone-Sax": "Sax",
    #     "Bass-Sax": "Sax",
    #     "Contrabass-Sax": "Sax",
    #     "Horn": "Horn",
    #     "Harmonica": "Remove",
    #     "Piccolo-Trumpet-Bb": "Trumpet",
    #     "Piccolo-Trumpet-A": "Trumpet",
    #     "High-Trumpet-F": "Trumpet",
    #     "High-Trumpet-Eb": "Trumpet",
    #     "High-Trumpet-D": "Trumpet",
    #     "Cornet": "Trumpet",
    #     "Trumpet": "Trumpet",
    #     "Trumpet-C": "Trumpet",
    #     "Trumpet-Bb": "Trumpet",
    #     "Cornet-Bb": "Trumpet",
    #     "Alto-Trumpet-F": "Trumpet",
    #     "Bass-Trumpet-Eb": "Trumpet",
    #     "Bass-Trumpet-C": "Trumpet",
    #     "Bass-Trumpet-Bb": "Trumpet",
    #     "Clarion": "Trumpet",
    #     "Trombone": "Trombone",
    #     "Alto-Trombone": "Trombone",
    #     "Soprano-Trombone": "Trombone",
    #     "Tenor-Trombone": "Trombone",
    #     "Bass-Trombone": "Trombone",
    #     "Contrabass-Trombone": "Trombone",
    #     "Euphonium": "Remove",
    #     "Tuba": "Tuba",
    #     "Bass-Tuba": "Tuba",
    #     "Contrabass-Tuba": "Tuba",
    #     "Flugelhorn": "Remove",
    #     "Piano": "Piano",
    #     "Celesta": "Remove",
    #     "Organ": "Organ",
    #     "Harpsichord": "Harpsichord",
    #     "Accordion": "Accordion",
    #     "Bandoneone": "Remove",
    #     "Harp": "Harp",
    #     "Guitar": "Guitar",
    #     "Bandurria": "Guitar",
    #     "Mandolin": "Guitar",
    #     "Lute": "Remove",
    #     "Lyre": "Remove",
    #     "Strings": "Remove",
    #     "Violin": "Violin",
    #     "Violins": "Violin",
    #     "Viola": "Viola",
    #     "Violas": "Viola",
    #     "Viola de gamba": "Viola",
    #     "Viola de braccio": "Remove",
    #     "Violoncello": "Violoncello",
    #     "Violoncellos": "Violoncello",
    #     "Contrabass": "Contrabass",
    #     "Basso continuo": "Remove",
    #     "Bass drum": "Remove",
    #     "Glockenspiel": "Remove",
    #     "Xylophone": "Remove",
    #     "Vibraphone": "Remove",
    #     "Marimba": "Remove",
    #     "Maracas": "Remove",
    #     "Bass-Marimba": "Remove",
    #     "Tubular-Bells": "Remove",
    #     "Clave": "Remove",
    #     "Bombo": "Remove",
    #     "Hi-hat": "Remove",
    #     "Triangle": "Remove",
    #     "Ratchet": "Remove",
    #     "Drum": "Remove",
    #     "Snare drum": "Remove",
    #     "Steel drum": "Remove",
    #     "Tambourine": "Remove",
    #     "Tam tam": "Remove",
    #     "Timpani": "Remove",
    #     "Cymbal": "Remove",
    #     "Castanets": "Remove",
    #     "Percussion": "Remove",
    #     "Voice": "Voice",
    #     "Voice soprano": "Voice",
    #     "Voice mezzo": "Voice",
    #     "Voice alto": "Voice",
    #     "Voice contratenor": "Voice",
    #     "Voice tenor": "Voice",
    #     "Voice baritone": "Voice",
    #     "Voice bass": "Voice",
    #     "Ondes martenot": "Remove",
    #     "Unknown": "Remove",
    # }

    #################################################
    #################################################
    #################################################
    #################################################
    # Violin : G3-C7
    # Viola : C3-E6
    # Cello : C2-C6
    # D-bass : C1-C4
    # VERSION HARDCORE QUATUOR
    simplify_mapping = {
        "Piccolo": "Violin",
        "Flute": "Violin",
        "Alto-Flute": "Violin",
        "Soprano-Flute": "Violin",
        "Bass-Flute": "Viola",
        "Contrabass-Flute": "Violoncello",
        "Pan Flute": "Remove",
        "Recorder": "Violin",
        "Ocarina": "Remove",
        "Oboe": "Violin",
        "Oboe-dAmore": "Violin",
        "Oboe de Caccia": "Remove",
        "English-Horn": "Viola",
        "Heckelphone": "Remove",
        "Piccolo-Clarinet-Ab": "Violin",
        "Clarinet": "Violin",
        "Clarinet-Eb": "Violin",
        "Clarinet-Bb": "Violin",
        "Piccolo-Clarinet-D": "Violin",
        "Clarinet-C": "Violin",
        "Clarinet-A": "Violin",
        "Basset-Horn-F": "Contrabass",
        "Alto-Clarinet-Eb": "Violoncello",
        "Bass-Clarinet-Bb": "Contrabass",
        "Bass-Clarinet-A": "Contrabass",
        "Contra-Alto-Clarinet-Eb": "Contrabass",
        "Contrabass-Clarinet-Bb": "Contrabass",
        "Bassoon": "Contrabass",
        "Contrabassoon": "Contrabass",
        "Soprano-Sax": "Violin",
        "Alto-Sax": "Viola",
        "Tenor-Sax": "Violoncello",
        "Baritone-Sax": "Violoncello",
        "Bass-Sax": "Contrabass",
        "Contrabass-Sax": "Contrabass",
        "Horn": "Contrabass",
        "Harmonica": "Remove",
        "Piccolo-Trumpet-Bb": "Violin",
        "Piccolo-Trumpet-A": "Violin",
        "High-Trumpet-F": "Violin",
        "High-Trumpet-Eb": "Violin",
        "High-Trumpet-D": "Violin",
        "Cornet": "Viola",
        "Trumpet": "Viola",
        "Trumpet-C": "Viola",
        "Trumpet-Bb": "Viola",
        "Cornet-Bb": "Viola",
        "Alto-Trumpet-F": "Viola",
        "Bass-Trumpet-Eb": "Violoncello",
        "Bass-Trumpet-C": "Violoncello",
        "Bass-Trumpet-Bb": "Violoncello",
        "Clarion": "Unknow",
        "Trombone": "Violin and Viola and Violoncello and Contrabass",
        "Alto-Trombone": "Violoncello",
        "Soprano-Trombone": "Viola",
        "Tenor-Trombone": "Contrabass",
        "Bass-Trombone": "Contrabass",
        "Contrabass-Trombone": "Trombone",
        "Euphonium": "Remove",
        "Tuba": "Contrabass",
        "Bass-Tuba": "Contrabass",
        "Contrabass-Tuba": "Contrabass",
        "Flugelhorn": "Remove",
        "Piano": "Piano",
        "Celesta": "Remove",
        "Organ": "Violin and Viola and Violoncello and Contrabass",
        "Harpsichord": "Violin and Viola and Violoncello and Contrabass",
        "Accordion": "Remove",
        "Bandoneone": "Remove",
        "Harp": "Remove",
        "Guitar": "Remove",
        "Bandurria": "Remove",
        "Mandolin": "Remove",
        "Lute": "Remove",
        "Lyre": "Remove",
        "Strings": "Violin and Viola and Violoncello and Contrabass",
        "Violin": "Violin",
        "Violins": "Violin",
        "Viola": "Viola",
        "Violas": "Viola",
        "Viola de gamba": "Viola",
        "Viola de braccio": "Remove",
        "Violoncello": "Violoncello",
        "Violoncellos": "Violoncello",
        "Contrabass": "Contrabass",
        "Basso continuo": "Remove",
        "Bass drum": "Remove",
        "Glockenspiel": "Remove",
        "Xylophone": "Remove",
        "Vibraphone": "Remove",
        "Marimba": "Remove",
        "Maracas": "Remove",
        "Bass-Marimba": "Remove",
        "Tubular-Bells": "Remove",
        "Clave": "Remove",
        "Bombo": "Remove",
        "Hi-hat": "Remove",
        "Triangle": "Remove",
        "Ratchet": "Remove",
        "Drum": "Remove",
        "Snare drum": "Remove",
        "Steel drum": "Remove",
        "Tambourine": "Remove",
        "Tam tam": "Remove",
        "Timpani": "Remove",
        "Cymbal": "Remove",
        "Castanets": "Remove",
        "Percussion": "Remove",
        "Voice": "Violin and Viola and Violoncello and Contrabass",
        "Voice soprano": "Violin",
        "Voice mezzo": "Viola",
        "Voice alto": "Viola",
        "Voice contratenor": "Viola",
        "Voice tenor": "Violoncello",
        "Voice baritone": "Violoncello",
        "Voice bass": "Contrabass",
        "Ondes martenot": "Remove",
        "Unknown": "Remove",
    }
    #################################################
    #################################################
    #################################################
    #################################################

    instru_name_unmixed = unmixed_instru(instru_name_complex)
    instru_name_unmixed_simple = []
    for e in instru_name_unmixed:
        simple_name = simplify_mapping[e]
        instru_name_unmixed_simple.append(simple_name)
    link = " and "
    return link.join(instru_name_unmixed_simple)


if __name__ == '__main__':
    name = 'DEBUG/test.mid'
    reader = Read_midi(name, 12)
    time = reader.get_time()
    pr = reader.read_file()
