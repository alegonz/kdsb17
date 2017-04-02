import os
from glob import glob
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


def random_rotation(x):
    """Random rotation of a 3D array.
    48 unique rotations = 6 (sides) x 4 (rotations of each side) x 2 (original and mirror).
    
    Args:
        x (numpy.array): 3D array to be rotated
    
    Returns:
        Randomly rotated array
    """
    # Convention: dim0 = z, dim1 = y, dim2=x

    if x.ndim != 3:
        raise ValueError('Array must be 3D. Got %d dimensions.' % x.ndim)

    # 3D array faces table
    # Rotations are either around the zy or zx planes.
    faces = [
        ((0, 1), 0),  # keep front face
        ((0, 1), 1),  # get bottom face front
        ((0, 1), 2),  # get back face front
        ((0, 1), 3),  # get top face front
        ((0, 2), -1),  # get right face front
        ((0, 2), 1)  # get left face front
    ]

    # Get a face front randomly
    axes1, turns1 = faces[np.random.choice(range(6))]
    xr = np.rot90(x, k=turns1, axes=axes1)

    # Rotate that face randomly (rotation around xy plane)
    turns2 = np.random.choice([0, 1, 2, 3])
    axes2 = (1, 2)
    xr = np.rot90(xr, k=turns2, axes=axes2)

    # Mirror the array randomly
    mirror = np.random.choice([True, False])
    if mirror:
        xr = np.fliplr(xr)

    return xr.copy()


def build_generator(data_path, labels_path,
                    mean=0, rescale_range=(-1000, 400),
                    rotate_randomly=True, random_offset_range=(-100, 100)):
    """Data generator for keras model.
    Args:
        data_path (str): path to npz files with array data.
        labels_path (str): path to csv file with labels.
        mean (float): Value for mean subtraction (scalar in HU units).
        rescale_range (tuple of float): Range of values in (in HU units) that will be mapped to [0, 1] respectively.
        rotate_randomly (bool): Rotate randomly arrays for data augmentation.
        random_offset_range (tuple): Random offset range (in HU units) for data augmentation (None=no random offset).
    
    Returns:
         A generator instance.
    """
    # TODO: add support for batch sizes greater than 1.
    # This is accomplished by grouping arrays of similar size and padding each appropriately to a common size.

    paths = glob(os.path.join(data_path, '*.npz'))
    labels = read_labels(labels_path, header=True)

    while 1:
        np.random.shuffle(paths)

        for path in paths:
            # Input data
            with np.load(path) as data:
                x = data['array_lungs']

            x = x.astype('float32')

            x -= mean

            if random_offset_range:
                x += np.random.uniform(random_offset_range[0], random_offset_range[1])

            if rotate_randomly:
                x = random_rotation(x)

            min_val, max_val = rescale_range
            x = (x - min_val) / (max_val - min_val)

            x = np.expand_dims(np.expand_dims(x, 0), 0)  # Add batch and channels dimensions (Theano format)

            # Output data
            patient_id, __ = os.path.splitext(os.path.basename(path))
            y = np.array([labels[patient_id]])

            yield (x, y)
