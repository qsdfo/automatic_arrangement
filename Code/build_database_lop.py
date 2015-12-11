import sys
import numpy as np
import music21 as m21
import math
import os
import json
import re
import time
# Imports de debug
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pylab as plt


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs


def build_db(database, quantization, instru_dict_path=None):

    # First load the instrument dictionnary
    if instru_dict_path is None:
        instru_dict = {}
    elif os.path.isfile(instru_dict_path):
        with open(instru_dict_path) as f:
            instru_dict = json.load(f)
    else:
        instru_dict = {}

    # Each pianoroll for each instrument is stored in a dictionnary indexed by
    # the name of the instruments
    pianoroll = {}

    # Keep a record of the transition between two tracks
    transition = []
    global_time = 0

    # Browse database folder
    for dirname, dirnames, filenames in os.walk(database):
        for filename in filenames:
            # Is it a music xml file ?
            filename_test = re.match("(.*)\.xml", filename, re.I)
            if not filename_test:
                continue

            # For each XML file, read it and store de results in a huge matrix
            score = m21.converter.parse(os.path.join(dirname, filename))
            # score = m21.corpus.parse('/Users/leo/anaconda/envs/Theano_2/lib/python2.7/site-packages/music21/corpus/beethoven/opus18no1/movement2.xml')
            duration_score = 0
            piano_roll_part_dico = {}
            piano_roll_part = {}
            # Loop over the different parts of the score
            for part in score.parts:

                ################################################
                # Get the name of the instrument if exists.
                # Add it to the dictionnary if not.
                ################################################
                part_name = part.id.decode(sys.stdin.encoding)
                is_found_instru = False

                for instru_name, set_name in instru_dict.iteritems():
                    # Match by regular expression
                    if match_re_list(part_name, set_name):
                        # Check if part name match any of the value in set_name
                        this_instru_name = instru_name
                        is_found_instru = True
                        break

                if not is_found_instru:
                    print '\n##########################################################'
                    print 'Enter an instrument name for the part name ' + part_name
                    print '(Be careful to use the same name as those shown before if this instrument already exists)\n'
                    print '\n'.join(instru_dict.keys())
                    print '@@@'
                    # Get the name of the instrument
                    this_instru_name = raw_input().decode(sys.stdin.encoding)

                    print 'Enter an associated regular expression :'
                    print '############################################################\n'
                    re_instru = raw_input().decode(sys.stdin.encoding)
                    # Add the name of the part to the list corresponding to this instrument in the dictionary
                    if this_instru_name in instru_dict.keys():
                        instru_dict[this_instru_name].append(re_instru)
                    else:
                        instru_dict[this_instru_name] = [re_instru]

                ################################################
                # D E B U G     P R I N T    P A R T
                # save_dir = 'PART_TXT_DEBUG/' + filename_test.group(1)
                # if not os.path.isdir(save_dir):
                #     os.mkdir(save_dir)
                # with open(save_dir + '/' + part_name + '.txt','w') as f:
                #     for el in part.recurse():
                #         print >>f, el
                ################################################
                ################################################
                ################################################

                ################################################
                # Get the pianoroll of this part
                ################################################
                # import pdb; pdb.set_trace()
                piano_roll_part = get_pianoroll_part(part, quantization)

                ################################################
                # Get the dynamcis
                ################################################
                dyn = get_dynamics(part, quantization)

                ################################################
                # Apply the dynamics on the pianoroll, if exists
                ################################################
                if dyn is not None:
                    current_dyn = 0
                    for time_frame in xrange(0, piano_roll_part.shape[1]):
                        current_dyn_list = [item[1] for item in dyn if item[0] == time_frame]
                        if current_dyn_list:
                            # There should be only one dynamic at a given time,
                            # but if there are 2 or more, take the first
                            current_dyn = current_dyn_list[0]
                        piano_roll_part[:, time_frame] = piano_roll_part[:, time_frame] * current_dyn

                ################################################
                # Add to the current part the pianoroll
                ################################################
                if this_instru_name in piano_roll_part_dico.keys():
                    # There is possibly many part associated with one instrument
                    # Hence max allows to deal with the two possible case :
                    #       1 - Exact same part splitted between two staves. We don't want to multiply by two
                    #               its dynamics, which would b the case if we simply add them.
                    #       2 - Two completely different parts, played by the same instrument at different staves.
                    #               We don't want to divide by 2 the dynamics by taking the mean of
                    #               those 2 parts.
                    size_part_1 = piano_roll_part_dico[this_instru_name].shape[1]
                    size_part_2 = piano_roll_part.shape[1]
                    if size_part_1 == size_part_2:
                        piano_roll_part_dico[this_instru_name] = np.maximum(piano_roll_part_dico[this_instru_name], piano_roll_part)
                    elif size_part_1 > size_part_2:
                        fill_size = size_part_1 - size_part_2
                        piano_roll_part_dico[this_instru_name] = np.maximum(piano_roll_part_dico[this_instru_name],
                                                                            + np.concatenate((piano_roll_part,
                                                                                              np.zeros((128, fill_size))),
                                                                                             axis=1))
                    else:
                        fill_size = size_part_2 - size_part_1
                        piano_roll_part_dico[this_instru_name] = np.maximum(np.concatenate((piano_roll_part_dico[this_instru_name], np.zeros((128, fill_size))), axis=1),
                                                                            + piano_roll_part)
                else:
                    piano_roll_part_dico[this_instru_name] = piano_roll_part

                ################################################
                ################################################
                ################################################
                ################################################
                #     D E B U G    P L O T
                ################################################
                # fig = plt.figure()
                # plt.matshow(piano_roll_part, cmap=plt.cm.gray, vmin = 0, vmax = 1)
                # plt.gca().invert_yaxis()
                # # plt.show()
                # save_dir = 'PDF_DEBUG/' + filename_test.group(1)
                # if not os.path.isdir(save_dir):
                #     os.mkdir(save_dir)
                # with PdfPages(save_dir + '/' + part_name + '.pdf') as pp:
                #     pp.savefig()
                # plt.close()
                ################################################
                ################################################
                ################################################
                ################################################

            ################################################
            # Fill with zeros each part
            # And add it to the global pianoroll
            ################################################

            # Should be the same at each iteration. If not... it's a problem
            duration_score = piano_roll_part.shape[1]

            for this_instru_name in instru_dict:
                # Deal with zero size part
                if this_instru_name in piano_roll_part_dico.keys():
                    # Case where the current instrument has been used in the currently parsed score
                    if this_instru_name in pianoroll.keys():
                            pianoroll[this_instru_name] = \
                                np.concatenate(
                                    (pianoroll[this_instru_name],
                                     piano_roll_part_dico[this_instru_name]),
                                    axis=1)
                    else:
                        if global_time is 0:
                            pianoroll[this_instru_name] = piano_roll_part_dico[this_instru_name]
                        else:
                            pianoroll[this_instru_name] = \
                                np.concatenate(
                                    (np.zeros((128, global_time)),
                                     piano_roll_part_dico[this_instru_name]),
                                    axis=1)
                else:
                    if this_instru_name in pianoroll.keys():
                        pianoroll[this_instru_name] = \
                            np.concatenate(
                                (pianoroll[this_instru_name],
                                 np.zeros((128, duration_score))),
                                axis=1)
                    else:
                        # Should never happen except if a score is written with empty staves...
                        if global_time is 0:
                            pianoroll[this_instru_name] = \
                                np.zeros((128, duration_score))
                        else:
                            pianoroll[this_instru_name] = \
                                np.concatenate(
                                    (np.zeros((128, global_time)),
                                     np.zeros((128, duration_score))),
                                    axis=1)

            global_time = global_time + duration_score
            transition.append(global_time + 1)

    ################################################
    # Save the instrument dictionary with its,
    # potentially, new notations per instrument
    ################################################
    if instru_dict_path is None:
        instru_dict_path = 'instrument_dico.json'
    save_data_json(instru_dict, instru_dict_path)
    return


def save_data_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=3, separators=(',', ': '))


def get_pianoroll_part(part, quantization):
    # Get the measure offsets
    measure_offset = {}
    for el in part.recurse(classFilter=('Measure')):
        measure_offset[el.measureNumber] = el.offset

    # Get the duration of the part
    duration_max = 0
    for el in part.recurse(classFilter=('Note', 'Rest')):
        t_end = get_end_time(el, measure_offset, quantization)
        if(t_end > duration_max):
            duration_max = t_end

    # First get the pitch and offset+duration
    piano_roll_part = np.zeros((128, int(math.ceil(duration_max))))
    for this_note in part.recurse(classFilter=('Note')):
        note_start = get_start_time(this_note, measure_offset, quantization)
        note_end = get_end_time(this_note, measure_offset, quantization)
        piano_roll_part[this_note.midi, note_start:note_end] = 1

    return piano_roll_part


def get_dynamics(part, quantization):
    measure_offset = {}
    for el in part.recurse(classFilter=('Measure')):
        measure_offset[el.measureNumber] = el.offset

    # Then the "static" dynamic stored in a list of tuples
    # Default dynamic is 0.5 at the beginning of the piece
    dyn = [(0, 0.5)]
    for this_dyn in part.recurse(classFilter=('Dynamic')):
        dyn_start = int(math.ceil((measure_offset[this_dyn.measureNumber] + this_dyn.offset) * quantization))
        dyn.append((dyn_start, this_dyn.volumeScalar))
    # Sort the list by its first argument (time)
    dyn = list(sorted(dyn, key=lambda item: item[0]))

    # Now the spanning dynamics (cresc, decresc)
    for this_spanning_dyn in part.recurse(classFilter=('Crescendo', 'Diminuendo', 'TextExpression')):
        if(type(this_spanning_dyn) is m21.expressions.TextExpression):
            txt = this_spanning_dyn.content
            if(re.match("cresc.*", txt, re.I) or re.match("dim.*", txt, re.I) or re.match("decr.*", txt, re.I)):
                start_time = get_start_time(this_spanning_dyn, measure_offset, quantization)
                if start_time is None:
                    # Avoid this element for which time information is missing
                    continue
                # Start dynamic is the previous defined dynamic
                for dyn_tuple in dyn:
                    if(dyn_tuple[0] > start_time):
                        break
                    start_dyn = dyn_tuple[1]
                end_time = start_time
                # Allow for a bigger threshold
                threshold = 8 * quantization
                end_dyn = start_dyn
                # Position keep track of the insert position for the current spanning event
                position = 0
                for dyn_tuple in dyn:
                    if(dyn_tuple[0] >= end_time):
                        if(dyn_tuple[0] < end_time + threshold):
                            end_dyn = dyn_tuple[1]
                            # We look at a long time span, but don't want the actual cresc/decresc to be that long
                            end_time = min(dyn_tuple[0], start_time + 4 * quantization)
                        break
                    position = position + 1

                # Deal with events spanning over a single note
                spanning_duration = int(end_time - start_time)
                if(re.match("cresc.*", txt, re.I)):
                    if(start_dyn >= end_dyn):
                        # In case we don't know where the crescendo goes,
                        # we are careful and increment the dynamics by only 0.2
                        # which is approximateley going from pp to p
                        end_dyn = min(0.9, start_dyn + 0.2)
                if(re.match("dim.*", txt, re.I) or re.match("decr.*", txt, re.I)):
                    if(start_dyn <= end_dyn):
                        end_dyn = max(0.1, start_dyn - 0.2)
                dyn_span = np.linspace(start_dyn, end_dyn, spanning_duration)
                # Will overlap by one frame on the next event, which is good
                for index in xrange(1, spanning_duration):
                    time = start_time + index
                    dyn.insert(position, (time, dyn_span[index]))
                    position = position + 1
        # Crescendo or diminuendo
        else:
            start_note = this_spanning_dyn.getFirst()
            end_note = this_spanning_dyn.getLast()

            start_time = get_start_time(start_note, measure_offset, quantization)
            end_time = get_end_time(end_note, measure_offset, quantization)

            # Avoid this element for which time information is missing
            if (start_time is None) or (end_time is None):
                continue

            # Start dynamic is the previous defined dynamic
            for dyn_tuple in dyn:
                if(dyn_tuple[0] > start_time):
                    break
                start_dyn = dyn_tuple[1]

            # End dynamics
            # We don't look further that the next quarter note (1)
            threshold = 1 * quantization
            end_dyn = start_dyn
            # Position keep track of the insert position for the current spanning event
            position = 0
            for dyn_tuple in dyn:
                if(dyn_tuple[0] >= end_time):
                    if(dyn_tuple[0] < end_time + threshold):
                        end_dyn = dyn_tuple[1]
                    break
                position = position + 1

            # Deal with events spanning over a single note
            spanning_duration = int(max(end_time - start_time, math.ceil((start_note.duration.quarterLength) * quantization)))

            # Do an interpolation, depending on the next dynamic if defined, a certain value if not
            if(type(this_spanning_dyn) == m21.dynamics.Crescendo):
                if(start_dyn >= end_dyn):
                    # In case we don't know where the crescendo goes,
                    # we are careful and increment the dynamics by only 0.2
                    # which is approximateley going from pp to p
                    end_dyn = min(0.9, start_dyn + 0.2)
            if(type(this_spanning_dyn) == m21.dynamics.Diminuendo):
                if(start_dyn <= end_dyn):
                    end_dyn = max(0.1, start_dyn - 0.2)

            dyn_span = np.linspace(start_dyn, end_dyn, spanning_duration)
            # Will overlap by one frame on the next event, which is good
            for index in xrange(1, spanning_duration):
                time = start_time + index
                dyn.insert(position, (time, dyn_span[index]))
                position = position + 1

    return dyn


def get_start_time(el, measure_offset, quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil((measure_offset[el.measureNumber] + el.offset) * quantization))
    # Else, no time defined for this element and the functino return None


def get_end_time(el, measure_offset, quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil((measure_offset[el.measureNumber] + el.offset + el.duration.quarterLength) * quantization))
    # Else, no time defined for this element and the functino return None


def match_re_list(expression, list):
    for value in list:
        result_re = re.match(expression, value, re.IGNORECASE)
        if result_re is not None:
            return True
    return False


if __name__ == '__main__':
    build_db('../Database/LOP_db_musescore/', 1, '../Database/LOP_db_musescore/instrument_dico.json')
    # build_db('../Database/LOP_db_small/', 1, '../Database/LOP_db_small/instrument_dico.json')
    # cProfile.run("build_db('../Database/LOP_db_small/', 4,'../Database/LOP_db_small/instrument_dico.json')")
