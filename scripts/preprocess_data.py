import os
import sys

import numpy as np

sys.path.append('/data/code/')

from kdsb17.preprocessing import (
    read_dcm_sequence, check_sequence, make_3d_array, resample, make_lungs_mask, extract_lungs
    )

from kdsb17.utils.plot import show_slices

base_path = '/data/data/stage1/'
out_path = '/data/analysis/temp/'

new_spacing = (1, 1, 1)  # (2, 2, 2)
kernel_size = 5  # 3
slice_drop_prob = (0.005, 0.005)
save_png = True

print('New spacing =', new_spacing)
print('Kernel size =', kernel_size)
print('Slice drop probability =', slice_drop_prob)

patients = [os.path.basename(path) for path in os.listdir(base_path)]

template = "An exception of type {0} occurred. Arguments:\n{1!r}"

for patient_id in patients:
    
    try:
        # Get slice sequence
        dcm_seq = read_dcm_sequence(patient_id, base_path)
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
