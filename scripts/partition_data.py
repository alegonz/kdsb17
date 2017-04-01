import os
import sys
import random

sys.path.append('/data/code/')

from kdsb17.trainutils import read_labels


def make_symlinks(file_names, src_path, dst_path):
    """Make symbolic links.
    Args:
        file_names: list of file names.
        src_path: path containing the source files.
        dst_path: path where the symlinks will be saved.
    """

    for file in file_names:
        os.symlink(os.path.join(src_path, file),
                   os.path.join(dst_path, file))


base_path = '/data/data/'
data_path = os.path.join(base_path, 'npz')
in_sample_csv_path = os.path.join(base_path, 'stage1_labels.csv')
out_of_sample_csv_path = os.path.join(base_path, 'stage1_sample_submission.csv')
train_ratio = 0.8  # the rest is for validation

# Read label data.
in_sample_labels = read_labels(in_sample_csv_path, header=True)
out_of_sample_labels = read_labels(out_of_sample_csv_path, header=True)

# Training and validation sets
patients = list(in_sample_labels.keys())
n_train = int(len(patients) * train_ratio)

random.seed(7102)
random.shuffle(patients)

train = patients[:n_train]
validation = patients[n_train:]

# Test set
test = list(out_of_sample_labels.keys())

# Make symbolic links
make_symlinks([patient_id + '.npz' for patient_id in train],
              data_path, os.path.join(base_path, 'train'))

make_symlinks([patient_id + '.npz' for patient_id in validation],
              data_path, os.path.join(base_path, 'validation'))

make_symlinks([patient_id + '.npz' for patient_id in test],
              data_path, os.path.join(base_path, 'test'))