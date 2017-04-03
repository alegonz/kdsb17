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


class GeneratorFactory:
    """Data generator for keras model.
    Args:
        data_path (str): path to npz files with array data.
        labels_path (str): path to csv file with labels.
        mean (float): Value for mean subtraction (scalar in HU units).
        rescale_map (tuple of tuple): HU values to normalized values, linear mapping pairs.
            Format: ((hu1, s1), (hu2, s2)) means that the mapping will be hu1 --> s1 and hu2 --> s2.
        rotate_randomly (bool): Rotate randomly arrays for data augmentation.
        random_offset_range (None or tuple): Random offset range (in HU units) for data augmentation (None=no random offset).

    Returns:
         A generator instance.
    """

    def __init__(self, data_path, labels_path,
                 mean=-350, rescale_map=((-1000, -1), (400, 1)),
                 rotate_randomly=True, random_offset_range=(-60, 60)):

        (hu1, s1), (hu2, s2) = rescale_map
        if (hu2 - hu1) == 0:
            raise ValueError('Invalid rescale mapping.')

        self.data_path = data_path
        self.paths = sorted(glob(os.path.join(data_path, '*.npz')))

        self.labels_path = labels_path
        self.labels = read_labels(labels_path, header=True)

        self.mean = mean
        self.rescale_map = rescale_map
        self.rotate_randomly = rotate_randomly
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

        if self.rotate_randomly:
            x = random_rotation(x)

        # Mean subtraction and rescaling
        x -= self.mean

        (hu1, s1), (hu2, s2) = self.rescale_map
        m = (s2 - s1) / (hu2 - hu1)
        x = m * (x - hu1) + s1

        x = np.expand_dims(np.expand_dims(x, 0), 0)  # Add batch and channels dimensions (Theano format)

        return x

    def create(self):
        """Data generator for keras model. This function reads from file and feed into the model one sample at a time.

        Args:
            None.

        Returns:
             A generator instance.
        """

        while 1:

            np.random.shuffle(self.paths)

            for path in self.paths:
                # Input data
                with np.load(path) as data:
                    x = data['array_lungs']

                x = self._transform(x)

                # Output data
                patient_id, __ = os.path.splitext(os.path.basename(path))
                label = self.labels[patient_id]
                y = np.array([label])

                yield (x, y)

    def create_on_memory(self, chunk_size=100):
        """Data generator for keras model. This function first uploads a chunk of samples into memory
        and then feeds those samples to the model.
        
        Args:
            chunk_size (int): number of samples to keep in memory at a time.
            
        Returns:
             A generator instance.
        """
        nb_chunks = len(self.paths) // chunk_size

        sizes = [chunk_size]*nb_chunks
        if len(self.paths) % chunk_size != 0:
            sizes.append(len(self.paths) % chunk_size)  # the last chunk will have less samples

        while 1:

            np.random.shuffle(self.paths)

            for chunk, size in enumerate(sizes):

                # Load chunk into memory
                n1 = chunk*chunk_size
                n2 = chunk*chunk_size + size

                samples = []
                for path in self.paths[n1:n2]:

                    # model input
                    with np.load(path) as data:
                        x = data['array_lungs']

                    x = self._transform(x)

                    # model output
                    patient_id, __ = os.path.splitext(os.path.basename(path))
                    label = self.labels[patient_id]
                    y = np.array([label])

                    samples.append((x, y))

                for x, y in samples:
                    yield (x, y)
