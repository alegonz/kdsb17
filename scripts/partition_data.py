import os
import sys
import random

sys.path.append('/data/code/')

from kdsb17.trainutils import read_labels


def makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


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


data_path = '/data/data/'
dataset = 'npz_2mm_ks3_05p'

all_path = os.path.join(data_path, dataset, 'all')
in_sample_csv_path = os.path.join(data_path, 'stage1_labels.csv')
out_of_sample_csv_path = os.path.join(data_path, 'stage1_sample_submission.csv')

train_ratio = 0.8  # the rest is for validation

subsets = {}

# Read label data.
in_sample_labels = read_labels(in_sample_csv_path, header=True)
out_of_sample_labels = read_labels(out_of_sample_csv_path, header=True)

# Training and validation sets
patients = list(in_sample_labels.keys())
n_train = int(len(patients) * train_ratio)

random.seed(7102)
random.shuffle(patients)

subsets['train'] = patients[:n_train]
subsets['validation'] = patients[n_train:]

# Test set
subsets['test'] = list(out_of_sample_labels.keys())

# Make symbolic links
for name, patient_list in subsets.items():

    path = makedir(os.path.join(data_path, dataset, name))
    relative_path = '../all'

    make_symlinks([patient_id + '.npz' for patient_id in patient_list],
                  relative_path, path)
