#!/usr/bin/env python3

import os
import time

from kdsb17.model import GaussianMixtureCAE
from kdsb17.utils.datagen import GeneratorFactory


def main():

    # --------- Data parameters
    checkpoints_path = '/work/data/kdsb17/analysis/checkpoints/'
    dataset_path = '/work/data/kdsb17/analysis/datasets/stage1/npz_spacing1x1x1_kernel5_drop0.5p/'

    # --------- Model parameters
    # Network parameters
    input_shape = (32, 32, 32)
    n_gaussians = 2
    nb_filters_per_layer = (64, 128, 256)
    kernel_size = (3, 3, 3)
    batch_normalization = False
    optimizer = 'adam'
    es_patience = 10

    # Training parameters
    batch_size = 32
    steps_per_epoch = 350
    epochs = 50
    validation_steps = 80
    chunk_size = 100

    # Define model
    time_string = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(checkpoints_path, time_string)  # dir for model files and log

    gmcae = GaussianMixtureCAE(input_shape=input_shape, n_gaussians=n_gaussians,
                               nb_filters_per_layer=nb_filters_per_layer, kernel_size=kernel_size,
                               batch_normalization=batch_normalization,
                               optimizer=optimizer, es_patience=es_patience, model_path=model_path)
    gmcae.build_model()

    # Create data generators
    train_gen_factory = GeneratorFactory(random_rotation=True, random_offset_range=None)
    val_gen_factory = GeneratorFactory(random_rotation=False, random_offset_range=None)

    train_gen = train_gen_factory.build_gmcae_generator(dataset_path, 'train', input_shape=input_shape,
                                                        batch_size=batch_size, chunk_size=chunk_size)

    val_gen = val_gen_factory.build_gmcae_generator(dataset_path, 'validation', input_shape=input_shape,
                                                    batch_size=batch_size, chunk_size=chunk_size)

    # Train model
    gmcae.fit_generator(train_generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_generator=val_gen, validation_steps=validation_steps)

if __name__ == '__main__':
    main()
