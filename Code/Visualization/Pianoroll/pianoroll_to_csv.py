#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
import os
import csv

# To save a numpy array :
# cPickle.dump(data, open(path_to_data + 'data.p', 'wb'))
# OR  numpy.save(open(path_to_data + 'data.p', 'wb'), data)

def pianoroll_to_csv(path_to_data, save_path='CSV_pianoroll'):
    # Load data
    data = cPickle.load(open(path_to_data, 'rb'))

    # Create directory
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # Quantization
    quantization = data['quantization']

    for score in data['scores'].itervalues():  # Or write the name of the file you want to see
        roll_all_instru = score['pianoroll']
        filename = score['filename']
        for instrument, pianoroll in roll_all_instru.iteritems():
            notes = pr_to_csv_aux(pianoroll, quantization, filename)
            save_dir = save_path + '/' + score['filename'] + '/'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            with open(save_dir + '/' + instrument + '.csv', 'wb') as f_handle:
                writer = csv.writer(f_handle, delimiter=',')
                writer.writerows(notes)


def pr_to_csv_aux(pianoroll, quantization=4, filename='unnamed'):
    # List of notes
    notes = []
    # pianoroll is a numpy array
    note_on = False
    dyn = 0
    for pitch in range(0, pianoroll.shape[1]):
        for time in range(0, pianoroll.shape[0]):
            dyn = pianoroll[time, pitch]
            if(dyn > 0):
                if not note_on:
                    # Beginning of the note
                    t0 = time
                    p = pitch
                    dyn_note = dyn
                    note_on = True
                elif(dyn != dyn_note):
                    # Change in dynamic
                    dt = time - t0
                    notes.append([t0, dt, p, dyn_note, quantization, filename])
                    t0 = time
                    p = pitch
                    dyn_note = dyn
                    note_on = True  # Not necessary...
            else:
                if note_on:
                    # End of a note
                    dt = time - t0
                    notes.append([t0, dt, p, dyn_note, quantization, filename])
                    note_on = False

    # Sort according to t0, for debug purposes
    notes = sorted(notes, key=lambda note: note[0])
    notes.insert(0, ['t0', 'dt', 'pitch', 'dyn', 'quantization', 'filename'])
    return notes

if __name__ == '__main__':
    pianoroll_to_csv('aaa.p')
