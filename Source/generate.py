#!/usr/bin/env python
# -*- coding: utf8 -*-

import cPickle as pkl
import reconstruct_pr
from acidano.data_processing.midi.write_midi import write_midi


def generate(model, piano, orchestra, indices, seed_size):
    # Generate sequences from a trained model
    # piano, orchestra and index are data used to seed the generation
    # Note that generation length is fixed by the length of the piano input
    #
    # generation_length : length of the sequence generated, seed included

    generation_length = piano.get_value(borrow=True).shape[0]

    generate_sequence = model.get_generate_function(
        piano=piano, orchestra=orchestra,
        generation_length=generation_length,
        seed_size=seed_size,
        batch_generation_size=len(indices),
        name="generate_sequence")

    # Given last indices, generate a batch of sequences
    (generated_sequences,) = generate_sequence(indices)

    return generated_sequences
