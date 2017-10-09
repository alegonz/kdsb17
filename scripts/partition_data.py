#!/usr/bin/env python3

import os
import sys
import random

from kdsb17.utils.file import read_labels, write_labels


def main(argv=None):
    """Partition the data. This script takes exactly two arguments from the command line.

    Args:
        argv (list): A list with two strings:
            1) Path containing the labels files.
            2) Path to save the partitioned labels files.
    """

    if argv is None:
        try:
            args = sys.argv
        except:
            raise SystemError('Command line arguments not found.')
    else:
        args = argv

    if len(args) != 3:
        raise ValueError('This script takes exactly two arguments: the path containing the labels files,'
                         'and the path to save the partitioned labels files.')

    labels_path, out_path = sys.argv[1], sys.argv[2]

    print('Reading labels files from:', labels_path)
    print('Saving partitioned files in:', out_path)

    in_sample_csv_path = os.path.join(labels_path, 'stage1_labels.csv')
    out_of_sample_csv_path = os.path.join(labels_path, 'stage1_solution.csv')

    train_ratio = 0.8  # the rest is for validation

    # Read label data.
    in_sample_labels = read_labels(in_sample_csv_path, header=True)
    out_of_sample_labels = read_labels(out_of_sample_csv_path, header=True)

    # Training and validation sets
    patients = list(in_sample_labels.keys())
    n_train = int(len(patients) * train_ratio)

    random.seed(7102)
    random.shuffle(patients)
    train_set = patients[:n_train]
    validation_set = patients[n_train:]

    train_labels = {pid: label for pid, label in in_sample_labels.items() if pid in train_set}
    validation_labels = {pid: label for pid, label in in_sample_labels.items() if pid in validation_set}

    # The test set is the same as the out-of-sample set

    # Save the sets to csv files
    write_labels(train_labels, os.path.join(out_path, 'train.csv'), header=True)
    write_labels(validation_labels, os.path.join(out_path, 'validation.csv'), header=True)
    write_labels(out_of_sample_labels, os.path.join(out_path, 'test.csv'), header=True)

if __name__ == '__main__':
    main()
