#!/usr/bin/env python3

import os
import time

from kdsb17.model import LungNet
from kdsb17.utils.datagen import GeneratorFactory


def main(argv=None):

    # Data file parameters
    checkpoints_path = '/root/share/personal/data/kdsb17/analysis/checkpoints/'
    dataset_path = '/root/share/personal/data/kdsb17/analysis/datasets/npz_spacing1x1x1_kernel5_drop0.5p/'

    # Training parameters
    input_size = (32, 32, 32)
    batch_size = 32
    steps_per_epoch = 250
    epochs = 20
    validation_steps = 100
    chunk_size = 100
    optimizer = 'adam'

    # Define model
    time_string = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(checkpoints_path, time_string)  # dir for model files and log

    lungnet = LungNet(nb_filters_per_layer=(64, 96, 128), n_gaussians=2,
                      optimizer=optimizer, batch_normalization=False,
                      model_path=model_path)
    lungnet.build_submodel('cae3d')

    # Create data generators
    train_gen_factory = GeneratorFactory(random_rotation=True, random_offset_range=None)
    val_gen_factory = GeneratorFactory(random_rotation=False, random_offset_range=None)

    train_gen = train_gen_factory.build_cae3d_generator(dataset_path, 'train', input_size=input_size,
                                                        batch_size=batch_size, chunk_size=chunk_size)

    val_gen = val_gen_factory.build_cae3d_generator(dataset_path, 'validation', input_size=input_size,
                                                    batch_size=batch_size, chunk_size=chunk_size)

    # Train model
    lungnet.fit_submodel('cae3d', train_generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                         validation_generator=val_gen, validation_steps=validation_steps)

if __name__ == '__main__':
    main()
