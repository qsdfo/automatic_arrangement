#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
@author: Pierre Talbot & Mattia G. Bergomi
"""

import numpy as np
from dtw import dtw
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations


notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
pause = ['p']

# Wrap pitch value in a class `Pitch`, `next` is a member containing the next pitch (of type `Pitch`) to be played in the voice leading sequence. It is `None` if it is the end of the sequence.
class Pitch:
  def __init__(self, value, offset):
    self.value = value
    self.offset = offset
    self.source_offset = offset
  def __repr__(self):
    return "(" + str(self.value) + "," + str(self.offset) + "," + str(self.source_offset) + ")"

def merge_max_multiset(res, from_multiset):
  for (e,n) in from_multiset.items():
    res[e] = max(res[e], n)

def map_source_to_target(sources, targets, pitches, target_idx, test):
  for (s, t) in zip(sources, targets):
    if test(s,t):
      source = pitches[s]
      target = pitches[t]
      target_idx[source.source_offset] = target.offset
      target.offset += 1
      source.source_offset += 1
  #print(target_idx)

def is_unison(s, t):
  return s == t

def is_not_unison(s, t):
  return not is_unison(s, t)

def partial_permut(sources, targets):
  sources = [500 if p in pause else notes.index( p[0] ) + 12*octaves.index( p[1]) if '#' not in p else notes.index( p[0:2] ) + 12*octaves.index( p[2]) for p in sources ]
  targets = [500 if p in pause else notes.index( p[0] ) + 12*octaves.index( p[1]) if '#' not in p else notes.index( p[0:2] ) + 12*octaves.index( p[2]) for p in targets ]

  multi_sources = Counter(sources)
  multi_targets = Counter(targets)

  #print(multi_sources)

  v = Counter()
  merge_max_multiset(v, multi_sources)
  merge_max_multiset(v, multi_targets)

  counted_pitches = sorted(v.items())
  #print(counted_pitches)

  pitches = {}
  counting_sum = 0
  for p, n in counted_pitches:
    pitches[p] = Pitch(p, counting_sum)
    counting_sum += n
  #print(pitches)

  v = sorted(v.elements())
  #print(v)

  target_idx = {}
  map_source_to_target(sources, targets, pitches, target_idx, is_unison)
  #print("After unison")
  map_source_to_target(sources, targets, pitches, target_idx, is_not_unison)

  w = [-1 for i in range(len(v))]
  # print sources, targets,target_idx.items(),v
  # Create the matrix `M` where `M[i][j] == 1` if the pitch at index `i` leads to the pitch at index `j`.
  dim = len(v)
  paused = []
  M = np.zeros((dim, dim))
  for (s, t) in target_idx.items():
    if v[s] != 500 and v[t] != 500:
        M[s][t] = 1
    else:
        M[s][t] = -1
        paused.append([s,t])
    w[s] = t
  w = np.add(w , 1)

  if paused:
    Minor = M
    for p_ind in paused:
        Minor = np.delete(Minor , (p_ind[0]) , axis = 0)
        Minor = np.delete(Minor , (p_ind[1]) , axis = 1)
  else:
    Minor = M

  c = [np.count_nonzero(np.triu(Minor,1)) , np.count_nonzero(np.tril(Minor,-1)) ,np.count_nonzero(np.diag(Minor)), 0 , 0]
  cross = 0

  non_zero = np.nonzero(Minor)
  couple = zip(non_zero[0] , non_zero[1])


  # print couple
  for ((i,j),(k,l)) in combinations(couple,2):
    if (i<j and k>l) and (i<k and j>l) or (i<=j and k>l) and (i<k and j>l) or (i<j and k>=l) and (i<k and j>l):
        cross = cross + 1
    elif (i<k and j>l):
        cross = cross + 1
  c[3] = cross
  c[4] = len(paused)
  # return M, Minor, w , c
  return c

# sources = ['C3' ,'E3','G3']
# targets = ['F3','D3','p']
# print(partial_permut(sources, targets))


#==============================================================================
#C1 = [['C1' , 'E1'] , ['C1' , 'E1'] , ['C1' , 'E1' , 'G1'], ['C1' , 'E1' , 'G1'] , ['C1' , 'E1'] , ['C1' , 'E1'] , ['C1' , 'E1'] ,['C1' , 'E1' , 'G1'],['C1' , 'E1' , 'G1'] ,  ['E1' , 'G1'] , ['C1' , 'E1'] , ['G2' , 'G2' , 'C3'] , ['C3','D4', 'D5']]
#C2 = [['C1' , 'E1'] , ['E1' , 'C1'] , [ 'G1','C1' , 'E1'], ['C1' , 'G1' , 'E1'] , ['E1' , 'C1'] , ['D1' , 'F1'] , ['F1' , 'D1'], ['D1' , 'F1' , 'A1'] , ['D1' , 'A1' , 'F1'], ['B0' , 'B1'] , ['F1' , 'F1'], ['C3' , 'C3' , 'C3'], ['D4','C3', 'C3']]
#for i in range(len(C1)):
#    print 'Voice Leading:', C1[i],C2[i]
#    print(partial_permut(C1[i],C2[i]))
#==============================================================================
# sources = ['C3' ,'D4','D5']
# targets = ['C3','C3','C3']
#==============================================================================
# Ex1: crab canon
# V1 = ['D4', 'D4' ,'D4', 'D4', 'F4', 'F4', 'F4', 'F4','A4' ,'A4' ,'A4' ,'A4', 'A#4', 'A#4', 'A#4', 'A#4','C#4', 'C#4', 'C#4', 'C#4', 'C#4', 'C#4', 'A4', 'A4','A4', 'A4', 'G#4', 'G#4' , 'G#4', 'G#4', 'G4', 'G4','G4', 'G4' ,'F#4', 'F#4', 'F#4' 'F#4', 'F4', 'F4','F4' ,'F4', 'E4', 'E4', 'D#4' ,'D#4', 'D4', 'D4','C#4', 'C#4' ,'A3', 'A3' ,'D4', 'D4', 'G4', 'G4','F4' ,'F4', 'F4', 'F4' ,'E4', 'E4', 'E4', 'E4','D4' ,'D4' ,'D4' ,'D4' ,'F4' ,'F4' ,'F4', 'F4','A4' , 'G4', 'A4', 'D5', 'A4', 'F4', 'E4', 'F4','G4' ,'A4' ,'B4', 'C#5', 'D5', 'F4', 'G4' ,'A4','A#4,' 'E4', 'F4', 'G4', 'A4', 'G4', 'F4' ,'E4','F4', 'G4' ,'A4', 'A#4', 'C5', 'A#4', 'A4' ,'G4','A4' ,'A#4', 'C5', 'D5', 'D#5', 'C5', 'A#4', 'A4','B4', 'C#5', 'D5' ,'E5', 'F5', 'D5', 'C#5', 'B5','C#5', 'D5', 'E5' ,'F5', 'G5', 'E5', 'A4', 'E5', 'D5' ,'E5', 'F5' ,'G5' ,'F5', 'E5', 'D5', 'C#5','D5', 'D5', 'A4' ,'A4' ,'F4', 'F4', 'D4', 'D4']
V1 = ['D4', 'D4' ,'D4', 'D4', 'F4', 'F4', 'F4', 'F4','A4' ,'A4' ,'A4' ,'A4', 'A#4', 'A#4', 'A#4', 'A#4','C#4', 'C#4', 'C#4', 'C#4', 'p', 'p', 'A4', 'A4','A4', 'A4', 'G#4', 'G#4' , 'G#4', 'G#4', 'G4', 'G4','G4', 'G4' ,'F#4', 'F#4', 'F#4' 'F#4', 'F4', 'F4','F4' ,'F4', 'E4', 'E4', 'D#4' ,'D#4', 'D4', 'D4','C#4', 'C#4' ,'A3', 'A3' ,'D4', 'D4', 'G4', 'G4','F4' ,'F4', 'F4', 'F4' ,'E4', 'E4', 'E4', 'E4','D4' ,'D4' ,'D4' ,'D4' ,'F4' ,'F4' ,'F4', 'F4','A4' , 'G4', 'A4', 'D5', 'A4', 'F4', 'E4', 'F4','G4' ,'A4' ,'B4', 'C#5', 'D5', 'F4', 'G4' ,'A4','A#4,' 'E4', 'F4', 'G4', 'A4', 'G4', 'F4' ,'E4','F4', 'G4' ,'A4', 'A#4', 'C5', 'A#4', 'A4' ,'G4','A4' ,'A#4', 'C5', 'D5', 'D#5', 'C5', 'A#4', 'A4','B4', 'C#5', 'D5' ,'E5', 'F5', 'D5', 'C#5', 'B5','C#5', 'D5', 'E5' ,'F5', 'G5', 'E5', 'A4', 'E5', 'D5' ,'E5', 'F5' ,'G5' ,'F5', 'E5', 'D5', 'C#5','D5', 'D5', 'A4' ,'A4' ,'F4', 'F4', 'D4', 'D4']
V2 = list(reversed(V1))
# Ex2: alleluia
V11 = ['F4','G4','A4','G4','F4','G4','A#4','A4','G4','F4','G4','F4','D4','F4','G4','A4','G4','F4','G4','A#4','A4','G4','D4','C4','D4']
V21 = ['C4','D4','E4','F4','G4','G4','F4', 'E4','F4','G4','G4','G4','G4','A4','G4','C5','B4','A4','G4','F4', 'G4','G4','G4','F4','D4']
# # Ex3: dicant nunc judei
V12 = ['F4','G4','F4','E4','D4','F4','F4','E4','F4','D4','E4','F4','D4','D4','D4','A4','A4','G4','F4','G4','A4','A4','D4','F4','E4','C4','D4','F4','G4','A4','G4','D4','F4','E4','F4','G4','F4','E4','D4','D4','A3','C4','D4']
V22 = ['C4','E4','D4','C4','D4','C4','D4','E4','C4','E4','C4','A3','C4','D4','D4','A3','C4','D4','F4','C4','A3','C4','D4','A4','G4','E4','D4','F4','E4','D4','C4','D4','D4','E4','C4','E4','D4','C4','D4','G4','A4','F4','D4']

array = []
for i in range(len(V1)-1):
    C1 = [V1[i] , V2[i]]
    C2 = [V1[i+1] , V2[i+1]]
    vl = partial_permut(C1,C2)
    if vl[2] != 2:
    	print 'Voice Leading:', C1,C2
    	print(partial_permut(C1,C2))
    	array.append(vl)

array1 = []
for i in range(len(V11)-1):
    C1 = [V11[i] , V22[i]]
    C2 = [V11[i+1] , V22[i+1]]
    print 'Voice Leading:', C1,C2
    print(partial_permut(C1,C2))
    array1.append(partial_permut(C1,C2))
array2 = []
for i in range(len(V12)-1):
    C1 = [V12[i] , V22[i]]
    C2 = [V12[i+1] , V22[i+1]]
    print 'Voice Leading:', C1,C2
    print(partial_permut(C1,C2))
    array2.append(partial_permut(C1,C2))



# Compute paradigmatic complexity vectors and their multiplicity, print them as a multiset, compute multiplicity for each vector and give a 3d scatter of these arrays as
# cloud of massive points.
uniq_complex_groups = [list(t) for t in set(map(tuple, array))]
multiplicity = []
for p in uniq_complex_groups:
  mult = array.count(p)
  multiplicity.append(mult)
print multiplicity
print zip(uniq_complex_groups, multiplicity)

# print uniq_complex_groups
# #==============================================================================
# x = []
# y = []
# z = []
# for el in uniq_complex_groups:
#      x += [el[0]]
#      y += [el[1]]
#      z += [el[3]]
# #print x,y,z
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# s = [np.divide(1000*m, len(V1)) for m in multiplicity]

# ax.scatter(x, y, z, c='r', marker='o',s=s)

# ax.set_xlabel('#voices moving upward')
# ax.set_ylabel('#voices moving downward')
# ax.set_zlabel('#voice crossing')


# x1 = []
# y1 = []
# z1 = []
# for el in uniq_complex_groups:
#      x1 += [el[0]]
#      y1 += [el[1]]
#      z1 += [el[4]]
# #print x,y,z
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# s = [np.divide(1000*m, len(V1)) for m in multiplicity]

# ax2.scatter(x1, y1, z1, c='r', marker='o',s=s)

# ax2.set_xlabel('#voices moving upward')
# ax2.set_ylabel('#voices moving downward')
# ax2.set_zlabel('#rests')

# plt.show()

# A little handmade thing
dist, cost, path = dtw(array, array1)
dist1, cost1, path1 = dtw(array, array2)
dist2, cost2, path2 = dtw(array1, array2)

print dist, dist1, dist2

plt.imshow(cost.T, origin='lower', cmap='bone', interpolation='bicubic')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0] - 0.5))
plt.ylim((-0.5, cost.shape[1] - 0.5))
plt.show()

plt.imshow(cost1.T, origin='lower', cmap='bone', interpolation='bicubic')
plt.plot(path1[0], path1[1], 'w')
plt.xlim((-0.5, cost1.shape[0] - 0.5))
plt.ylim((-0.5, cost1.shape[1] - 0.5))
plt.show()

plt.imshow(cost2.T, origin='lower', cmap='bone', interpolation='bicubic')
plt.plot(path2[0], path2[1], 'w')
plt.xlim((-0.5, cost2.shape[0] - 0.5))
plt.ylim((-0.5, cost2.shape[1] - 0.5))
plt.show()
