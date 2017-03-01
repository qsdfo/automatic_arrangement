#!/usr/bin/env python
# -*- coding: utf8 -*-

import cPickle as pkl
from reconstruct_pr import reconstruct_pr, reconstruct_piano
from acidano.data_processing.midi.write_midi import write_midi


def generate(model,
             piano, orchestra, indices, metadata_path,
             generation_length, seed_size, quantization_write, temporal_granularity,
             generated_folder, logger_generate):
    # Generate sequences from a trained model
    # piano, orchestra and index are data used to seed the generation
    # Note that generation length is fixed by the length of the piano input
    logger_generate.info("# Generating")

    generate_sequence = model.get_generate_function(
        piano=piano, orchestra=orchestra,
        generation_length=generation_length,
        seed_size=seed_size,
        batch_generation_size=len(indices),
        name="generate_sequence")

    # Load the mapping between pitch space and instrument
    metadata = pkl.load(open(metadata_path, 'rb'))
    instru_mapping = metadata['instru_mapping']

    # Given last indices, generate a batch of sequences
    (generated_sequence,) = generate_sequence(indices)

    # Get the ground-truth piano and orchestra
    # IZY from piano, orchestra, indices, generation length and seed_size
    piano_seed = model.build_seed(piano.get_value(), indices, len(indices), generation_length)
    orchestra_original = model.build_seed(orchestra.get_value(), indices, len(indices), generation_length)
    if generated_folder is not None:
        for write_counter in xrange(generated_sequence.shape[0]):

            ###############################################################
            ###############################################################
            ###############################################################
            ###############################################################
            if temporal_granularity == 'event_level':
                quantization_write = 1
            ###############################################################
            ###############################################################
            ###############################################################
            ###############################################################

            # Write generated midi
            pr_orchestra = reconstruct_pr(generated_sequence[write_counter], instru_mapping)
            write_path = generated_folder + '/' + str(write_counter) + '_generated.mid'
            write_midi(pr_orchestra, quantization_write, write_path, tempo=80)

            # Write original piano
            pr_piano_seed = reconstruct_piano(piano_seed[write_counter], instru_mapping)
            write_path = generated_folder + '/' + str(write_counter) + '_piano_seed.mid'
            write_midi(pr_piano_seed, quantization_write, write_path, tempo=80)

            # Write original orchestra
            pr_orchestra_original = reconstruct_pr(orchestra_original[write_counter], instru_mapping)
            write_path = generated_folder + '/' + str(write_counter) + '_orchestra_original.mid'
            write_midi(pr_orchestra_original, quantization_write, write_path, tempo=80)

    return
