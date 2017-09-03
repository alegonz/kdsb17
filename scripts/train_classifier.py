#!/usr/bin/env python3

import os
import sys
import time

from kdsb17.utils.datagen import GeneratorFactory
from kdsb17.model import LungNet


def main():

    # ---------- Data parameters
    checkpoints_path = '/work/data/kdsb17/analysis/checkpoints/'
    gmcae_weights_path = '20170903_060745/weights.17-0.432386.hdf5'
    data_path = '/work/data/kdsb17/analysis/datasets/stage1/'
    dataset = 'npz_spacing1x1x1_kernel5_drop0.5p'

    # ---------- Model parameters
    # Encoder parameters. These must be match the ones used in GaussianMixtureCAE.
    nb_filters_per_layer = (64, 128, 256)
    kernel_size = (3, 3, 3)
    batch_normalization = False

    # Classifier parameters
    n_dense = (1024, 1024)
    dropout_rate = 0.5

    optimizer = 'adam'
    es_patience = 10

    freeze = ['a', 'b']

    # Training parameters
    # batch_size is 1 (full stochastic)
    steps_per_epoch = 350
    epochs = 50
    validation_steps = 80

    # Define model
    time_string = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(checkpoints_path, time_string)  # dir for model files and log

    lungnet = LungNet(nb_filters_per_layer=nb_filters_per_layer, kernel_size=kernel_size,
                      batch_normalization=batch_normalization,
                      n_dense=n_dense, dropout_rate=dropout_rate,
                      optimizer=optimizer, es_patience=es_patience,
                      model_path=model_path)

    lungnet.build_model(freeze=freeze)

    lungnet.load_weights_from_file(os.path.join(checkpoints_path, gmcae_weights_path))

    # Create data generators
    train_gen_factory = GeneratorFactory(random_rotation=True, random_offset_range=None)
    val_gen_factory = GeneratorFactory(random_rotation=False, random_offset_range=None)

    dataset_path = os.path.join(data_path, dataset)
    train_gen = train_gen_factory.build_classifier_generator(dataset_path, 'train')
    val_gen = val_gen_factory.build_classifier_generator(dataset_path, 'validation')

    # Train model
    lungnet.fit_generator(train_generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                          validation_generator=val_gen, validation_steps=validation_steps)


if __name__ == '__main__':
    sys.exit(main())
