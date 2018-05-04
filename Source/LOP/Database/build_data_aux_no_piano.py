#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:05:13 2017

@author: leo
"""

import numpy as np
import os
import csv
import re

from build_data_aux import simplify_instrumentation

from LOP_database.utils.event_level import get_event_ind_dict
from LOP_database.utils.time_warping import warp_pr_aux
from LOP_database.utils.pianoroll_processing import sum_along_instru_dim
from LOP_database.utils.pianoroll_processing import get_pianoroll_time

import Musicxml_parser.scoreToPianoroll as mxml
from LOP_database.midi.read_midi import Read_midi

def get_instru_and_pr_from_folder_path_NP(folder_path, quantization):
    # There should be 1 files
    music_file = [e for e in os.listdir(folder_path) if re.search(r'\.(mid|xml)$', e)]
    assert len(music_file)==1, "There should be only one music file"
    music_file = music_file[0]
    music_file_path = os.path.join(folder_path, music_file)

    # Get the type
    if re.search(r'\.mid$', music_file):
        # Deduce csv files
        csv_file_path = re.sub(r'\.mid$', '.csv', music_file_path)
        # Get pr    
        reader_midi = Read_midi(music_file_path, quantization)
        pianoroll = reader_midi.read_file()
    else:
        csv_file_path = re.sub(r'\.xml$', '.csv', music_file_path)
        pianoroll, articulation = mxml.scoreToPianoroll(music_file_path, quantization)
        
    total_time = get_pianoroll_time(pianoroll)

    try:
        with open(csv_file_path, 'r') as ff:
            rr = csv.DictReader(ff, delimiter=';')
            instru = next(rr)
    except:
        import pdb; pdb.set_trace()

    # Simplify names : keep only tracks not marked as useless
    instru_simple = {k: simplify_instrumentation(v) for k, v in instru.items()}
    # Files name, no extensions
    name = re.sub(r'\.(mid|csv)$', '', music_file)

    return pianoroll, instru_simple, total_time, name
    
    
def process_folder_NP(folder_path, quantization, temporal_granularity):
    """Get the pianoroll from a folder path with containing only an orchestral score. 
    Piano score is created by simply crushing all the instruments on 88 pitches
    """
    # Get instrus and prs from a folder name name
    pr_orch, instru_orch, T, name = get_instru_and_pr_from_folder_path_NP(folder_path, quantization)

    # Temporal granularity
    if temporal_granularity == 'event_level':
        event_orch = get_event_ind_dict(pr_orch)
        T = len(event_orch)
        def get_duration(event, last_time):
            start_ind = event[:]
            end_ind = np.zeros(event.shape, dtype=np.int)
            end_ind[:-1] = event[1:]
            end_ind[-1] = last_time
            duration_list = end_ind - start_ind
            return duration_list
        duration_orch = get_duration(event_orch, T)
        # Get the duration of each event
        pr_orch = warp_pr_aux(pr_orch, event_orch)
    else:
        event_orch = None
    
    # Create the piano score
    pr_piano = {'Piano': sum_along_instru_dim(pr_orch)}
    event_piano = event_orch
    duration_piano = duration_orch
    instru_piano = {'Piano': 'Piano'}
    name_piano = name
    
    return pr_piano, event_piano, duration_piano, instru_piano, name_piano, pr_orch, event_orch, duration_orch, instru_orch, name, T