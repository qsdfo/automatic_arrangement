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
import pdb
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import mpldatacursor


def build_db(database, quantization, instru_dict_path=None):
    # First load the instrument dictionnary
    if instru_dict_path is None:
        # Create a defaut empty file if not indicated
        instru_dict_path = database + u"instrument_dico.json"
        instru_dict = {}
    elif os.path.isfile(instru_dict_path):
        with open(instru_dict_path) as f:
            instru_dict = json.load(f)
    else:
        raise NameError(instru_dict_path + " is not a json file")

    # Each pianoroll for each instrument is stored in a dictionnary indexed by
    # the name of the instruments
    pianoroll = {}
    articulation = {}
    dynamics = {}

    # Keep a record of the transition between two tracks
    transition = []
    counter = 0
    data = {}
    global_time = 0

    # Browse database folder
    for dirname, dirnames, filenames in os.walk(database):
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
                             'dynamics': Handler_score.dynamics,
                             'filename': filename}
            counter += 1

            # track_length = Handler_score.pianoroll['Piano'].shape[0]
            # pianoroll_keys = set(pianoroll.keys())
            #
            # for instru in Handler_score.pianoroll.keys():
            #     this_pianoroll = Handler_score.pianoroll[instru]
            #     this_articulation = Handler_score.articulation[instru]
            #     this_dynamics = Handler_score.dynamics[instru]
            #     ################################################
            #     ################################################
            #     ################################################
            #     # Debug plot
            #     f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
            #     ax1.imshow(np.transpose(this_pianoroll), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=1)
            #     ax1.invert_yaxis()
            #     ax2.imshow(np.transpose(this_articulation), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=1)
            #     ax2.invert_yaxis()
            #     ax3.plot(this_dynamics)
            #     # mpldatacursor.datacursor(display='single')
            #     # plt.show()
            #     save_dir = 'PDF_DEBUG/' + filename_test.group(1)
            #     if not os.path.isdir(save_dir):
            #         os.mkdir(save_dir)
            #     with PdfPages(save_dir + '/' + instru + '.pdf') as pp:
            #         pp.savefig()
            #     plt.close(f)
            #
            #     # Dump debug
            #     non_ze_pianoroll = np.nonzero(this_pianoroll)
            #     te_pianoroll = np.concatenate((non_ze_pianoroll[0], non_ze_pianoroll[1])).reshape(2, non_ze_pianoroll[0].shape[0])
            #     non_ze_articulation = np.nonzero(this_articulation)
            #     te_articulation = np.concatenate((non_ze_articulation[0], non_ze_articulation[1])).reshape(2, non_ze_articulation[0].shape[0])
            #     save_dir = 'DUMP_MATRICES/' + filename_test.group(1)
            #     if not os.path.isdir(save_dir):
            #         os.mkdir(save_dir)
            #     with open(save_dir + '/' + instru + '_pianoroll.csv', 'w') as f_handle:
            #         np.savetxt(f_handle, te_pianoroll, delimiter=";", fmt='%1i')
            #     with open(save_dir + '/' + instru + '_articulation.csv', 'w') as f_handle:
            #         np.savetxt(f_handle, te_articulation, delimiter=";", fmt='%1i')
            #     ################################################
            #     ################################################
            #     ################################################
            #     if instru in pianoroll_keys:
            #         pianoroll[instru] = np.concatenate((pianoroll[instru], this_pianoroll))
            #         articulation[instru] = np.concatenate((articulation[instru], this_articulation))
            #         dynamics[instru] = np.concatenate((dynamics[instru], this_dynamics))
            #         pianoroll_keys.remove(instru)
            #     else:
            #         # Fill with zeros the beginnig of this newly instanciated instrument
            #         pianoroll[instru] = np.concatenate((np.zeros((global_time, 128)), this_pianoroll))
            #         articulation[instru] = np.concatenate((np.zeros((global_time, 128)), this_articulation))
            #         dynamics[instru] = np.concatenate((np.zeros((global_time)), this_dynamics))
            #
            # # Scan through the instrument missing in this particular score
            # # And fill their pianorolls with the necessary number of zeros
            # for instru in pianoroll_keys:
            #     pianoroll[instru] = np.concatenate((pianoroll[instru], np.zeros((track_length, 128))))
            #     articulation[instru] = np.concatenate((articulation[instru], np.zeros((track_length, 128))))
            #     dynamics[instru] = np.concatenate((dynamics[instru], np.zeros((track_length))))
            #
            # # Increment the time counter
            # global_time += track_length
            # transition.append(global_time + 1)

    ################################################
    # Save the instrument dictionary with its,
    # potentially, new notations per instrument
    ################################################
    if instru_dict_path is None:
        instru_dict_path = 'instrument_dico.json'
    save_data_json(instru_dict, instru_dict_path)
    pickle.dump(data, open('data.p', 'wb'))
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
    build_db('../../../Database/LOP_db_small/', 4, '../../../Database/LOP_db_small/instrument_dico.json')

# data = Handler_score.pianoroll['Piano']
# non_ze = np.nonzero(data)
# te = np.concatenate((non_ze[0], non_ze[1])).reshape(2, non_ze[0].shape[0])
# with open('nonzeros.csv', 'w') as f_handle:
#     np.savetxt(f_handle, te, delimiter=";", fmt='%1i')

# f, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.imshow(np.transpose(Handler_score.pianoroll['Piano']), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=1)
# ax1.invert_yaxis()
# ax2.imshow(np.transpose(Handler_score.articulation['Piano']), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=1)
# plt.gca().invert_yaxis()
# # mpldatacursor.datacursor(display='single')
# plt.show()
