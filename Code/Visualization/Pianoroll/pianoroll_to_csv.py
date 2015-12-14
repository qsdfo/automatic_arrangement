#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
import os
import csv


def pianoroll_to_csv(path_to_data):
    # Load data
    data = cPickle.load(open(path_to_data, 'rb'))
    # Create directory
    if not os.path.isdir('CSV_pianoroll'):
        os.mkdir('CSV_pianoroll')

    for score in data.itervalues():  # Or write the name of the file you want to see
        roll_all_instru = score['pianoroll']
        for instrument, pianoroll in roll_all_instru.iteritems():
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
                            notes.append([t0, dt, p, dyn_note])
                            t0 = time
                            p = pitch
                            dyn_note = dyn
                            note_on = True  # Not necessary...
                    else:
                        if note_on:
                            # End of a note
                            dt = time - t0
                            notes.append([t0, dt, p, dyn_note])
                            note_on = False

            # Sort according to t0, for debug purposes
            notes = sorted(notes, key=lambda note: note[0])
            notes.insert(0, ['t0', 'dt', 'pitch', 'dyn'])
            save_dir = 'CSV_pianoroll/' + score['filename'] + '/'
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            with open(save_dir + '/' + instrument + '.csv', 'wb') as f_handle:
                writer = csv.writer(f_handle)
                writer.writerows(notes)


# def write_note(pianoroll_svg, x, dx, y, note_height, h_note):
#     g = pianoroll_svg << g(
#         transform="translate(" + x + "," + y ")")
#     g << rect(width=dx, height=note_height,
#               style="fill:rgb(" + h_note + "," + h_note + "," + h_note + ")")
#     x_text = x + 1
#     y_text = y + note_height / 2
#     text = g << text("T=" + x + ":" + dx + " P:" + pitch + " D:" + h_note,
#                      x=x_text, y=y_text,
#                      style="fill:rgb(0,255,0)", display="none",)
if __name__ == '__main__':
    pianoroll_to_csv('../../../Data/data.p')
