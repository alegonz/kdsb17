import os
from glob import glob
from itertools import product

import numpy as np

from kdsb17.utils.file import read_labels
from kdsb17.preprocessing import zoom

EPSILON = 1e-6


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

    def rotate3d(self, x, key):
        """Random rotation of a 3D array.
        48 unique rotations = 6 (sides) x 4 (rotations of each side) x 2 (original and mirror).

        Args:
            x (numpy.array): 3D array to be rotated
            key (int): Pattern key.

        Returns:
            Randomly rotated array
        """
        # Convention: dim0 = z, dim1 = y, dim2=x
        if x.ndim != 3:
            raise ValueError('Array must be 3D. Got %d dimensions.' % x.ndim)

        (axes1, k1), (axes2, k2), mirror = self[key]

        # Get the specified face front
        xr = np.rot90(x, k=k1, axes=axes1)
        # Turn that face
        xr = np.rot90(xr, k=k2, axes=axes2)
        # Mirror the array randomly
        if mirror:
            xr = np.fliplr(xr)

        return xr


class GeneratorFactory:
    """Data generator for model.
    Args:
        rescale_map (tuple of tuple): HU values to normalized values, linear mapping pairs.
            Format: ((hu1, s1), (hu2, s2)) means that the mapping will be hu1 --> s1 and hu2 --> s2.
        random_rotation (bool): Rotate randomly arrays for data augmentation.
        random_offset_range (None or tuple): Offset range (in HU units) for data augmentation (None=no random offset).
            Suggested value: (-60, 60)
    """

    def __init__(self,
                 rescale_map=((-1000, -1), (400, 1)), volume_resize_factor=None,
                 random_rotation=False, random_offset_range=None):

        (hu1, s1), (hu2, s2) = rescale_map
        if abs(hu2 - hu1) <= EPSILON or abs(s2 - s1) <= EPSILON:
            raise ValueError('Invalid rescale mapping.')

        self.rescale_map = rescale_map
        self.volume_resize_factor = volume_resize_factor
        self.random_rotation = random_rotation
        self._rotation_patterns = RotationPatterns48()
        self.random_offset_range = random_offset_range

    def _array2io(self, array):
        z, y, x = array.shape
        array = array.astype('float32')

        # Rescaling
        (hu1, s1), (hu2, s2) = self.rescale_map
        m = (s2 - s1) / (hu2 - hu1)
        array = m * (array - hu1) + s1

        i = np.reshape(array, (1, z, y, x, 1))  # Add sample and channels dimensions (Tensorflow format)
        o = np.reshape(array, (1, z*y*x))  # Flatten z,y,x dimensions and add sample dimension

        return i, o

    def _transform(self, array):
        """Transform sample into array to be fed into keras model. The transformation performs:
        - Volume resizing (useful for controlling memory usage)
        - Random rotation
        - Random offset addition
        """

        if self.volume_resize_factor:
            alpha = self.volume_resize_factor ** (1/3.0)
            array = zoom(array, zoom=alpha, mode='nearest')

        # Data augmentation
        if self.random_offset_range:
            array += np.random.uniform(low=self.random_offset_range[0],
                                       high=self.random_offset_range[1])

        if self.random_rotation:
            idx = np.random.randint(low=0, high=48)
            array = self._rotation_patterns.rotate3d(array, idx)

        return array

    @staticmethod
    def _get_subset_info(dataset_path, subset):

        array_paths = glob(os.path.join(dataset_path, '*.npz'))
        patient_ids = [os.path.basename(path).split('.')[0] for path in array_paths]

        try:
            labels = read_labels(os.path.join(dataset_path, subset + '.csv'), header=True)

        except Exception as e:
            raise IOError(e)

        else:
            not_found = [pid for pid in labels.keys() if pid not in patient_ids]

            if len(not_found) > 0:
                raise IOError('No npz data found for Patients:', not_found)

            return labels

    def build_classifier_generator(self, dataset_path, subset):
        """Data generator for binary classification model.
        Reads from file and feed into the model one sample at a time.

        Args:
            dataset_path (str): path to folder with array data (npz files) and label data (csv files).
            subset (str): Name of subset. Either 'train', 'validation' or 'test'.
        Yields:
             A tuple (x, y) of data array and label.
        """

        labels = self._get_subset_info(dataset_path, subset)
        patient_ids = list(labels.keys())

        while 1:

            np.random.shuffle(patient_ids)

            for patient_id in patient_ids:

                # Input data
                path = os.path.join(dataset_path, patient_id + '.npz')
                with np.load(path) as array_data:
                    x = array_data['array_lungs']

                x, _ = self._array2io(self._transform(x))

                # Output data
                y = labels.get(patient_id)
                if y is None:
                    raise ValueError('Unknown label for patient %s.' % patient_id)
                y = np.array([[y]], dtype='float32')

                yield (x, y)

    def build_gmcae_generator(self, dataset_path, subset,
                              input_shape, batch_size=32, chunk_size=150):
        """Data generator for 3d convolutional autoencoder.
        First uploads a chunk of samples into memory and then feeds subarrays in batches to the model one at time.

        Args:
            dataset_path (str): path to folder with array data (npz files) and label data (csv files).
            subset (str): Name of subset. Either 'train', 'validation' or 'test'.
            input_shape (tuple): size of input subarray in (z, y, x) order.
            batch_size (int): batch size.
            chunk_size (int): number of samples to keep in memory at a time.

        Yields:
             A tuple (x, x) of data subarrays.
        """

        labels = self._get_subset_info(dataset_path, subset)
        patient_ids = list(labels.keys())

        n_samples = len(patient_ids)

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
            np.random.shuffle(patient_ids)

            for n1, n2 in chunks:

                # Load chunk of patient arrays into memory
                arrays = []
                for patient_id in patient_ids[n1:n2]:

                    path = os.path.join(dataset_path, patient_id + '.npz')
                    with np.load(path) as array_data:
                        x = array_data['array_lungs']
                    arrays.append(x)

                # Extract samples from each array (these are just views)
                samples = []
                for array in arrays:
                    nb_bins_z, nb_bins_y, nb_bins_x = [n // s for n, s in zip(array.shape, input_shape)]

                    for i, j, k in product(range(nb_bins_z), range(nb_bins_y), range(nb_bins_x)):
                        # each combination of i,j,k is a sample
                        i1, i2 = i * input_shape[0], (i + 1) * input_shape[0]
                        j1, j2 = j * input_shape[1], (j + 1) * input_shape[1]
                        k1, k2 = k * input_shape[2], (k + 1) * input_shape[2]

                        samples.append(array[i1:i2, j1:j2, k1:k2])

                # Shuffle the samples
                np.random.shuffle(samples)

                # Yield batches
                x_in = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2], 1), dtype='float32')
                x_out = np.zeros((batch_size, input_shape[0]*input_shape[1]*input_shape[2]), dtype='float32')

                idx = 0
                for sample in samples:
                    x_in[idx % batch_size], x_out[idx % batch_size] = self._array2io(self._transform(sample))

                    idx += 1
                    if (idx % batch_size) == 0:
                        yield (x_in, x_out)
