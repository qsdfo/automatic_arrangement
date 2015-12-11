# MusicXML parser

## Description :
Music XML parser written in Python based on a SAX analyzer.
Given an mxml file, outputs three dictionaries indexed by instruments names. For each instrument, three Numpy arrays are created :
- a binary pianoroll (note on/off along time given a certain rhythmic quantization)
- a binary articulation which is the same matrix as the pianoroll but with shorter duration so that
the we can distinguish between a long note or several repeated occurences of the same note. Hence, if a quarter note lasted 4 frames in the pianoroll it would be 3 in the articulation. Staccati notes are 1 whatever their duration.
- a dynamic vector which indicate the evolution of the dynamic for this instrument along time

Written in python 2.7


## Packages dependencies :
- xml.sax
- Numpy
- os, re, json, sys

## Files
### mxml_parser :
Main function which parse a database and save the built pianoroll in a json format.
### scoreToPianoroll.py :
Implements the main
