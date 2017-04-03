import os
import sys
import random

sys.path.append('/data/code/')

from kdsb17.trainutils import read_labels
from kdsb17.fileutils import makedir, make_symlinks

data_path = '/data/data/'
dataset = 'npz_1mm_ks5_05p'

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
