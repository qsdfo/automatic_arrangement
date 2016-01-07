#!/usr/bin/env python
# -*- coding: utf8 -*-
# A SAX-based parser for a MusicXML file written for multi-instrument scores
# with dynamics
# This is a minimalist yet exhaustive parser.
# It produces as an output a matrix corresponding to the piano-roll representation
#
# The file has to be parsed two time. A first time to get the total duration of the file.
#
# Idea : a minus amplitude at the end of a note (play the role of a flag saying : note stop here)
#
# TODO : Ajouter les <articulations> (notations DOSIM quoi wech)

import xml.sax
import json
import os
import re
from scoreToPianoroll import ScoreToPianorollHandler
from totalLengthHandler import TotalLengthHandler
# Debug
# import pdb
import cPickle
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import mpldatacursor


def build_db(database_path, quantization, instru_dict_path=None, output_path='data.p'):
    # First load the instrument dictionnary
    if instru_dict_path is None:
        # Create a defaut empty file if not indicated
        instru_dict_path = database_path + u"instrument_dico.json"
        instru_dict = {}
    elif os.path.isfile(instru_dict_path):
        with open(instru_dict_path) as f:
            instru_dict = json.load(f)
    else:
        raise NameError(instru_dict_path + " is not a json file")

    # Data are stored in a dictionnary
    data = {}
    counter = 0

    # Browse database_path folder
    for dirname, dirnames, filenames in os.walk(database_path):
        for filename in filenames:
            # Is it a music xml file ?
            filename_test = re.match("(.*)\.xml", filename, re.I)
            if not filename_test:
                continue

            full_path_file = os.path.join(dirname, filename)

            print "Parsing file : " + filename

            # Get the total length in quarter notes of the track
            pre_parser = xml.sax.make_parser()
            pre_parser.setFeature(xml.sax.handler.feature_namespaces, 0)
            Handler_length = TotalLengthHandler()
            pre_parser.setContentHandler(Handler_length)
            pre_parser.parse(full_path_file)
            total_length = Handler_length.total_length

            # Now parse the file and get the pianoroll, articulation and dynamics
            parser = xml.sax.make_parser()
            parser.setFeature(xml.sax.handler.feature_namespaces, 0)
            Handler_score = ScoreToPianorollHandler(quantization, instru_dict, total_length, False)
            parser.setContentHandler(Handler_score)
            parser.parse(full_path_file)

            data[counter] = {'pianoroll': Handler_score.pianoroll,
                             'articulation': Handler_score.articulation,
                             'filename': filename,
                             'quantization': quantization}
            counter += 1

    ################################################
    # Save the instrument dictionary with its,
    # potentially, new notations per instrument
    ################################################
    if instru_dict_path is None:
        instru_dict_path = 'instrument_dico.json'
    save_data_json(instru_dict, instru_dict_path)
    cPickle.dump(data, open(output_path, 'wb'))
    # data = {}
    # data['pianoroll'] = pianoroll
    # data['articulation'] = articulation
    # data['dynamics'] = dynamics
    # (data, 'data.json')
    return


def save_data_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=3, separators=(',', ': '))


if __name__ == '__main__':
    build_db('../../../Database/LOP_db_small/', 4, '../../../Database/LOP_db_small/instrument_dico.json', '../../../Data/data.p')
