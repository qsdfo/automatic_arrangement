#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
import os
import csv
import numpy as np


def dump_to_csv(path_to_data='dump.csv', save_path='data.csv'):
    with open(path_to_data, 'rb') as f:
        mat = np.genfromtxt(f, delimiter=',', dtype=None)
        list_point = mat_to_csv(mat)
    with open(save_path, 'wb') as f_handle:
        writer = csv.writer(f_handle, delimiter=',')
        writer.writerows(list_point)


def mat_to_csv(mat):
    # List of rect_list
    points = []
    # mat is a numpy array
    z = 0
    for y in range(0, mat.shape[1]):
        for x in range(0, mat.shape[0]):
            z = mat[x, y]
            if z != 0:
                points.append([x, y, z])

    # Sort according to t0, for debug purposes
    points = sorted(points, key=lambda note: note[0])
    points.insert(0, ['x', 'y', 'z'])
    return points

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
    dump_to_csv('dump.csv')
