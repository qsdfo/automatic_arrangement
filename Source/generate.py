#!/usr/bin/env python
# -*- coding: utf8 -*-

import cPickle as pkl
from reconstruct_pr import reconstruct_pr
from acidano.data_processing.midi.write_midi import write_midi


def generate(model,
             piano, orchestra, indices, metadata_path,
             generation_length, seed_size, quantization_write,
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
    import pdb; pdb.set_trace()
    (generated_sequence,) = generate_sequence(indices)
    # Get the ground-truth piano and orchestra
    # IZY from piano, orchestra, indices, generation length and seed_size
    # orchestra_original = orchestra[indices]
    if generated_folder is not None:
        for write_counter in xrange(generated_sequence.shape[0]):
            # Write midi
            pr_orchestra = reconstruct_pr(generated_sequence[write_counter], instru_mapping)
            write_path = generated_folder + '/' + str(write_counter) + '.mid'
            write_midi(pr_orchestra, quantization_write, write_path, tempo=80)

    return
