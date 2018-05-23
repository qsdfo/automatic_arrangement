""" Implementation of DenseNet architecture for CNN.
from paper : https://arxiv.org/pdf/1608.06993.pdf

Slightly modified by : Mathieu Prang (mathieu.prangircam.fr)

Construct a Dense architecture especially tailored for symbolic musical data.
"""

import torch
import torch.nn as nn


# Add transition
def addTransition(model, nChannels, nOutChannels, dropout):
    model.batchnormT = nn.BatchNorm1d(nChannels)
    model.reluT = nn.ReLU()
    model.convT = nn.Conv1d(nChannels, nOutChannels, kernel_size=12, padding=6)
    if dropout > 0:
        model.dropoutT = nn.Dropout(dropout)
    model.poolingT = nn.MaxPool1d(2, ceil_mode=True, return_indices=True)


# Add back transition
def addTransitionBack(model, nChannels, nOutChannels, dropout):
    model.unpoolTB = nn.MaxUnpool1d(2)
    model.batchnormTB = nn.BatchNorm1d(nChannels)
    model.reluTB = nn.ReLU()
    model.convTB = nn.Conv1d(nChannels, nOutChannels, kernel_size=12, padding=6)
    if dropout > 0:
        model.dropoutTB = nn.Dropout(dropout)


# Add single layer
def singleLayer(model, nChannels, nOutChannels, dropout, i):
    setattr(model, 'batchnorm' + str(i), nn.BatchNorm1d(nChannels))
    setattr(model, 'relu' + str(i), nn.ReLU())
    setattr(model, 'conv' + str(i), nn.Conv1d(nChannels, nOutChannels, kernel_size=12, padding=6))
    if dropout > 0:
        setattr(model, 'dropout' + str(i), nn.Dropout(dropout))