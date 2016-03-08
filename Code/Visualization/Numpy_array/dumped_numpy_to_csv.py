#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle
import os
import csv
import numpy as np

# To save a numpy array in a csv file :
# np.savetxt('dump.csv', data_np, delimiter=',')

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


if __name__ == '__main__':
    dump_to_csv('orch.csv')
