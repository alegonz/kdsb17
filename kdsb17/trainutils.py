import os
from itertools import product
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


def get_crop_idx(input_size, output_size):
    """Get the crop sizes necessary to match the CAE output image size.

    Args:
        input_size  (tuple): input array size (z, y, x)
        output_size (tuple): output array size (z, y, x)
    Returns:
        crop indexes along each dimension (tuple): ((dim1_1, dim1_2), (dim2_1, dim2_2), ..., (dimN_1, dimN_2))
    """

    if any([i < o for i, o in zip(input_size, output_size)]):
        raise ValueError('The output size must be less than or equal to the input size.')

    idx = []

    for dim_in, dim_out in zip(input_size, output_size):
        diff = dim_in - dim_out
        idx1 = diff // 2
        idx2 = dim_in - (diff // 2 + diff % 2)

        idx.extend([(idx1, idx2)])

    return tuple(idx)


class RotationPatterns48:
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


class Generator3dCNN:
    """Data generator for 3D CNN.
    Args:
        data_path (str): path to npz files with array data.
        labels_path (str): path to csv file with labels.
        mean (float): Value for mean subtraction (scalar in HU units).
        rescale_map (tuple of tuple): HU values to normalized values, linear mapping pairs.
            Format: ((hu1, s1), (hu2, s2)) means that the mapping will be hu1 --> s1 and hu2 --> s2.
        random_rotation (bool): Rotate randomly arrays for data augmentation.
        random_offset_range (None or tuple): Offset range (in HU units) for data augmentation (None=no random offset).
            Suggested value: (-60, 60)

    Returns:
         A generator instance.
    """

    def __init__(self, data_path, labels_path,
                 mean=-346.65, rescale_map=((-1000, -1), (400, 1)),
                 random_rotation=False, random_offset_range=None):

        (hu1, s1), (hu2, s2) = rescale_map
        if (hu2 - hu1) == 0:
            raise ValueError('Invalid rescale mapping.')

        self.data_path = data_path
        self.patients = sorted(glob(os.path.join(data_path, '*.npz')))

        self.labels_path = labels_path
        self.labels = read_labels(labels_path, header=True)

        self.mean = mean
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
        - Change dimension to Theano format.
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

        # Mean subtraction
        x -= (m * (self.mean - hu1) + s1)

        x = np.expand_dims(np.expand_dims(x, 0), 0)  # Add batch and channels dimensions (Theano format)

        return x

    def for_binary_classifier(self):
        """Data generator for binary classification model.
        Reads from file and feed into the model one sample at a time.

        Args:
            None.

        Returns:
             A generator instance.
        """

        while 1:

            np.random.shuffle(self.patients)

            for patient in self.patients:
                # Input data
                with np.load(patient) as data:
                    x = data['array_lungs']

                x = self._transform(x)

                # Output data
                patient_id, __ = os.path.splitext(os.path.basename(patient))
                label = self.labels[patient_id]
                y = np.array([label])

                yield (x, y)

    def for_binary_classifier_chunked(self, chunk_size=100):
        """Data generator for binary classification model.
        First uploads a chunk of samples into memory and then feeds those samples to the model one at time.
        
        Args:
            chunk_size (int): number of samples to keep in memory at a time.
            
        Returns:
             A generator instance.
        """
        nb_chunks = len(self.patients) // chunk_size

        sizes = [chunk_size]*nb_chunks
        if len(self.patients) % chunk_size != 0:
            sizes.append(len(self.patients) % chunk_size)  # the last chunk will have less samples

        while 1:

            np.random.shuffle(self.patients)

            for chunk, size in enumerate(sizes):

                # Load chunk into memory
                n1 = chunk*chunk_size
                n2 = chunk*chunk_size + size

                samples = []
                for patient in self.patients[n1:n2]:

                    # model input
                    with np.load(patient) as data:
                        x = data['array_lungs']

                    x = self._transform(x)

                    # model output
                    patient_id, __ = os.path.splitext(os.path.basename(patient))
                    label = self.labels[patient_id]
                    y = np.array([label])

                    samples.append((x, y))

                for x, y in samples:
                    yield (x, y)

    def for_autoencoder_chunked(self, input_size, batch_size=32, chunk_size=150):

        # Calculate chunk indices
        nb_chunks = len(self.patients) // chunk_size

        chunks = []
        for chunk in range(nb_chunks):
            n1 = chunk_size * chunk
            n2 = chunk_size * (chunk + 1)
            chunks.append((n1, n2))

        if len(self.patients) % chunk_size != 0:
            n1 = chunk_size * nb_chunks
            n2 = n1 + len(self.patients) % chunk_size
            chunks.append((n1, n2))  # the last chunk will have less patients

        # Generator's endless loop starts here
        while 1:
            np.random.shuffle(self.patients)

            for n1, n2 in chunks:

                # Load chunk of patient arrays into memory
                arrays = []
                for patient in self.patients[n1:n2]:
                    with np.load(patient) as data:
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
                x = np.zeros((batch_size, 1, input_size[0], input_size[1], input_size[2]), dtype='float32')

                idx = 0
                for sample in samples:
                    x[idx % batch_size, 0] = self._transform(sample)
                    idx += 1

                    if (idx % batch_size) == 0:
                        yield (x, x)
