#!/usr/bin/env python3

import os
import sys

import numpy as np

from kdsb17.preprocessing import (read_dcm_sequence, check_sequence, make_3d_array,
                                  resample, make_lungs_mask, extract_lungs)
from kdsb17.utils.plot import show_slices
from kdsb17.utils.file import makedir


def main(argv=None):
    """Preprocess the dicom data. This script takes exactly two arguments from the command line.

    Args:
        argv (list): A list with two strings:
            1) Parent path containing the patient folders (dicom data).
            2) Output path to save the preprocessed data (npz array data).
    """

    if argv is None:
        try:
            args = sys.argv
        except:
            raise SystemError('Command line arguments not found.')
    else:
        args = argv

    if len(args) != 3:
        raise ValueError('This script takes exactly two arguments: the path containing the patient folders,'
                         'and the path to save the preprocessed data.')

    parent_path, out_path = sys.argv[1], sys.argv[2]

    new_spacing = (1, 1, 1)  # (2, 2, 2)
    kernel_size = 5  # 3
    slice_drop_prob = 0.005
    save_png = False
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"

    # Make output directory
    dataset_name = 'npz_spacing%s_kernel%d_drop%.1fp' % ('x'.join([str(s) for s in new_spacing]),
                                                         kernel_size,
                                                         100*slice_drop_prob)

    parent_name = os.path.basename(parent_path.rstrip('/'))
    out_path = os.path.join(out_path, parent_name, dataset_name)
    makedir(out_path)

    print('New spacing:', new_spacing)
    print('Kernel size:', kernel_size)
    print('Slice drop probability:', slice_drop_prob)
    print('Output path:', out_path)

    patients = [os.path.basename(path) for path in os.listdir(parent_path)]

    for patient_id in patients:

        try:
            # Get slice sequence
            dcm_seq = read_dcm_sequence(patient_id, parent_path)
            dcm_seq = check_sequence(dcm_seq)

            # Stack into 3D array in Hounsfield Units
            array, spacing = make_3d_array(dcm_seq)

            # Resample
            array_resampled = resample(array, spacing, new_spacing)

            # Extract lungs
            mask_lungs, thres = make_lungs_mask(array_resampled, kernel_size=kernel_size)
            array_lungs, box = extract_lungs(array_resampled, mask_lungs, slice_drop_prob)

            # Outputs
            if save_png:
                png_filename = os.path.join(out_path, patient_id + '.png')
                show_slices(array_lungs, filename=png_filename, every=5)

            npz_filename = os.path.join(out_path, patient_id + '.npz')
            np.savez_compressed(npz_filename, array_lungs=array_lungs)

        except Exception as ex:
            message = template.format(type(ex).__name__, ex.args)
            print('fail', patient_id, message)

            continue

        else:
            print(
                'success',
                patient_id, thres, box,
                array.shape, array_resampled.shape, array_lungs.shape
                )

if __name__ == '__main__':
    main()
