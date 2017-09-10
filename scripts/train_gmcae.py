#!/usr/bin/env python3

import os
import time
import numpy as np

from kdsb17.model import GaussianMixtureCAE
from kdsb17.utils.datagen import GeneratorFactory


def main():

    # --------- Data parameters
    checkpoints_path = '/root/share/personal/data/kdsb17/analysis/checkpoints/'
    dataset_path = '/root/share/personal/data/kdsb17/analysis/datasets/stage1/npz_spacing1x1x1_kernel5_drop0.5p/'

    # --------- Model parameters
    # Network parameters
    n_gaussians = 4
    input_shape = (32, 32, 32)
    nb_filters_per_layer = (64, 128, 256)
    kernel_size = (3, 3, 3)
    padding = 'same'
    batch_normalization = False
    optimizer = 'adam'
    es_patience = 10
    histogram_freq = 1

    # Training parameters
    batch_size = 32
    steps_per_epoch = 350
    epochs = 100
    validation_steps = 80
    chunk_size = 100

    # Define model
    time_string = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(checkpoints_path, time_string)  # dir for model files and log

    gmcae = GaussianMixtureCAE(n_gaussians=n_gaussians, input_shape=input_shape,
                               nb_filters_per_layer=nb_filters_per_layer, kernel_size=kernel_size, padding=padding,
                               batch_normalization=batch_normalization,
                               optimizer=optimizer, es_patience=es_patience, model_path=model_path,
                               histogram_freq=histogram_freq)
    gmcae.build_model()

    gmcae.summary()

    # Create data generators
    train_gen_factory = GeneratorFactory(random_rotation=True, random_offset_range=None)
    val_gen_factory = GeneratorFactory(random_rotation=False, random_offset_range=None)

    train_gen = train_gen_factory.build_gmcae_generator(dataset_path, 'train', input_shape=input_shape,
                                                        batch_size=batch_size, chunk_size=chunk_size)

    val_gen = val_gen_factory.build_gmcae_generator(dataset_path, 'validation', input_shape=input_shape,
                                                    batch_size=batch_size, chunk_size=chunk_size)

    # Train model
    if histogram_freq > 0:
        # The TensorBoard callback cannot make histograms if the validation data comes from a generator
        # Thus, we have to burn the generator to make a static validation set
        val_data_x = []
        val_data_y = []

        for i, (x, y) in enumerate(val_gen):
            if i == validation_steps:
                break
            val_data_x.append(x)
            val_data_y.append(y)

        val_data = (np.concatenate(val_data_x, axis=0),
                    np.concatenate(val_data_y, axis=0))
    else:
        val_data = val_gen

    gmcae.fit_generator(train_generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_generator=val_data, validation_steps=validation_steps)

if __name__ == '__main__':
    main()
