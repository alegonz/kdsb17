import os
from itertools import product
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

    labels = {patient_id: int(label) for patient_id, label in lines}

    return labels


def rotate3d(x, pattern):
    """Random rotation of a 3D array.
    48 unique rotations = 6 (sides) x 4 (rotations of each side) x 2 (original and mirror).

    Args:
        x (numpy.array): 3D array to be rotated
        pattern (tuple): an element of a RotationPatterns48 instance

    Returns:
        Randomly rotated array
    """
    # Convention: dim0 = z, dim1 = y, dim2=x
    if x.ndim != 3:
        raise ValueError('Array must be 3D. Got %d dimensions.' % x.ndim)

    (axes1, k1), (axes2, k2), mirror = pattern

    # Get the specified face front
    xr = np.rot90(x, k=k1, axes=axes1)
    # Turn that face
    xr = np.rot90(xr, k=k2, axes=axes2)
    # Mirror the array randomly
    if mirror:
        xr = np.fliplr(xr)

    return xr


class RotationPatterns48:
    """Class that creates a unique rotation of a 3D array, by combining face flips, turns and mirroring.
    There are 48 unique rotations in total (6 faces x 4 turns x 2 mirrors)
    """
    def __init__(self):
        # Rotations around the zy or zx planes.
        self.flips = [
            ((0, 1), 0),   # keep front face
            ((0, 1), 1),   # get bottom face front
            ((0, 1), 2),   # get back face front
            ((0, 1), 3),   # get top face front
            ((0, 2), -1),  # get right face front
            ((0, 2), 1)    # get left face front
        ]

        # Rotations around xy plane
        self.turns = [
            ((1, 2), 0),  # 0 degrees
            ((1, 2), 1),  # 90 degrees
            ((1, 2), 2),  # 180 degrees
            ((1, 2), 3)   # 270 degrees
        ]

        self.mirrors = [False, True]

    def __getitem__(self, key):
        key = key % 48

        flip = key // 8
        turn = (key % 8) // 2
        mirror = key % 2

        return self.flips[flip], self.turns[turn], self.mirrors[mirror]


class GeneratorFactory:
    """Data generator for model.
    Args:
        rescale_map (tuple of tuple): HU values to normalized values, linear mapping pairs.
            Format: ((hu1, s1), (hu2, s2)) means that the mapping will be hu1 --> s1 and hu2 --> s2.
        random_rotation (bool): Rotate randomly arrays for data augmentation.
        random_offset_range (None or tuple): Offset range (in HU units) for data augmentation (None=no random offset).
            Suggested value: (-60, 60)

    Returns:
         A generator instance.
    """

    def __init__(self, rescale_map=((-1000, -1), (400, 1)),
                 random_rotation=False, random_offset_range=None):

        (hu1, s1), (hu2, s2) = rescale_map
        if (hu2 - hu1) == 0:
            raise ValueError('Invalid rescale mapping.')

        self.rescale_map = rescale_map
        self.random_rotation = random_rotation
        self.rotation_patterns = RotationPatterns48()
        self.random_offset_range = random_offset_range

    def _transform(self, x):
        """Transform sample into array to be fed into keras model. The transformation performs:
        - Casting to float32
        - Random rotation
        - Random offset addition
        - Mean subtraction
        - Rescaling
        """

        x = x.astype('float32')

        # Data augmentation
        if self.random_offset_range:
            x += np.random.uniform(low=self.random_offset_range[0],
                                   high=self.random_offset_range[1])

        if self.random_rotation:
            idx = np.random.randint(low=0, high=48)
            x = rotate3d(x, self.rotation_patterns[idx])

        # Rescaling
        (hu1, s1), (hu2, s2) = self.rescale_map

        m = (s2 - s1) / (hu2 - hu1)
        x = m * (x - hu1) + s1

        x = np.expand_dims(np.expand_dims(x, 0), 4)  # Add batch and channels dimensions (Tensorflow format)

        return x

    def build_classifier_generator(self, data_paths, labels_path):
        """Data generator for binary classification model.
        Reads from file and feed into the model one sample at a time.

        Args:
            data_paths (str): path to npz files with array data.
            labels_path (str): path to csv file with labels.
        Yields:
             A tuple (x, y) of data array and label.
        """

        if len(data_paths) == 0:
            raise ValueError('data_paths cannot be an empty list.')

        labels = read_labels(labels_path, header=True)

        while 1:

            np.random.shuffle(data_paths)

            for path in data_paths:
                # Input data
                with np.load(path) as data:
                    x = data['array_lungs']

                x = self._transform(x)

                # Output data
                patient_id, __ = os.path.splitext(os.path.basename(path))

                y = labels.get(patient_id)
                if y is None:
                    raise ValueError('Unknown label for patient %s.' % patient_id)
                y = np.array([[y]], dtype='float32')

                yield (x, y)

    def build_autoencoder_generator(self, data_paths, input_size, batch_size=32, chunk_size=150):
        """Data generator for 3d convolutional autoencoder.
        First uploads a chunk of samples into memory and then feeds subarrays in batches to the model one at time.

        Args:
            data_paths (str): path to npz files with array data.
            input_size (tuple): size of input subarray in (z, y, x) order.
            batch_size (int): batch size.
            chunk_size (int): number of samples to keep in memory at a time.

        Yields:
             A tuple (x, x) of data subarrays.
        """

        if len(data_paths) == 0:
            raise ValueError('data_paths cannot be an empty list.')

        n_samples = len(data_paths)

        # Calculate chunk indices
        nb_chunks = n_samples // chunk_size

        chunks = []
        for chunk in range(nb_chunks):
            n1 = chunk_size * chunk
            n2 = chunk_size * (chunk + 1)
            chunks.append((n1, n2))

        if n_samples % chunk_size != 0:
            n1 = chunk_size * nb_chunks
            n2 = n1 + n_samples % chunk_size
            chunks.append((n1, n2))  # the last chunk will have less patients

        # Generator's endless loop starts here
        while 1:
            np.random.shuffle(data_paths)

            for n1, n2 in chunks:

                # Load chunk of patient arrays into memory
                arrays = []
                for path in data_paths[n1:n2]:
                    with np.load(path) as data:
                        x = data['array_lungs']
                    arrays.append(x)

                # Extract samples from each array (these are just views)
                samples = []
                for array in arrays:
                    nb_bins_z, nb_bins_y, nb_bins_x = [n//s for n, s in zip(array.shape, input_size)]

                    for i, j, k in product(range(nb_bins_z), range(nb_bins_y), range(nb_bins_x)):
                        # each combination of i,j,k is a sample
                        i1, i2 = i * input_size[0], (i + 1) * input_size[0]
                        j1, j2 = j * input_size[1], (j + 1) * input_size[1]
                        k1, k2 = k * input_size[2], (k + 1) * input_size[2]

                        samples.append(array[i1:i2, j1:j2, k1:k2])

                # Shuffle the samples
                np.random.shuffle(samples)

                # Yield batches
                x = np.zeros((batch_size, input_size[0], input_size[1], input_size[2], 1), dtype='float32')

                idx = 0
                for sample in samples:
                    x[idx % batch_size, 0] = self._transform(sample)
                    idx += 1

                    if (idx % batch_size) == 0:
                        yield (x, x)
