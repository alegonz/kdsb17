import os
from glob import glob
import random
import numpy as np


def read_labels(path, header=True):
    """Makes dictionary of patient_id:labels from csv file.
    Args:
        path: path to csv file containing the patient_id labels.
        header (bool): csv file has a header (True) or not (False).
    
    Returns:
         Dictionary of patient_id:label.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    if header:
        lines.pop(0)

    lines = [tuple(line.rstrip().split(',')) for line in lines]

    labels = {patient_id: float(label) for patient_id, label in lines}

    return labels


def build_generator(data_path, labels_path, mean=0, rescale_range=(-1000, 400), seed=2017):
    """Data generator for keras model.
    Args:
        data_path (str): path to npz files with array data.
        labels_path (str): path to csv file with labels.
        mean (float): Value for mean subtraction (scalar).
        rescale_range (tuple of float): Range of values in the original data that will be mapped to [0, 1] respectively.
        seed (int): Random seed for data file shuffling.
    
    Returns:
         A generator instance.
    """
    # TODO: add support for batch sizes greater than 1.
    # This is accomplished by grouping arrays of similar size and padding each appropriately to a common size.
    #
    # TODO: add random 48 rotations
    #
    # TODO: consider random addition of constant value

    random.seed(seed)

    paths = glob(os.path.join(data_path, '*.npz'))
    labels = read_labels(labels_path, header=True)

    while 1:
        random.shuffle(paths)

        for path in paths:
            # Input data
            with np.load(path) as data:
                x = data['array_lungs']

            x = x.astype('float32')

            # Subtract mean and normalize
            min_val, max_val = rescale_range
            x -= mean
            x = (x - min_val) / (max_val - min_val)

            # Add here random rotation
            # Add here addition of random constant

            # Add batch and channels dimensions
            # TODO: currently Theano only. Add support for TensorFlow dim ordering.
            x = np.expand_dims(np.expand_dims(x, 0), 0)

            # Output data
            patient_id, __ = os.path.splitext(os.path.basename(path))
            y = np.array([labels[patient_id]])

            yield (x, y)
