import os
import sys

import numpy as np

sys.path.append('/sandbox/data_science_bowl_2007/code/')

from preprocessing import (
    read_dcm_sequence, check_sequence, make_3d_array, resample, extract_lungs
    )

from plotutils import show_slices

base_path = '/Shanghai/kaggle_data_science_bowl_2017_data/stage1/'
out_path = '/sandbox/data_science_bowl_2007/analysis/temp/'

patients = [os.path.basename(path) for path in os.listdir(base_path)]

for patientid in patients:
    
    try:
        # Get slice sequence
        dcm_seq = read_dcm_sequence(patientid, base_path)
        check_sequence(dcm_seq)
        
        # Stack into 3D array in Hounsfield Units
        array, spacing = make_3d_array(dcm_seq)
        
        # Resample
        array_resampled = resample(array, spacing, [2,1,1])
        
        # Extract lungs
        array_lungs, thres, box = extract_lungs(array_resampled)
    
    except:
        print('x', patientid, ' failed at preprocessing.')
        continue
    
    # Outputs
    try:
        png_filename = os.path.join(out_path, patientid+'.png')
        show_slices(array_lungs, filename=png_filename, every=2)
        
        npz_filename = os.path.join(out_path, patientid+'.npz')
        np.savez_compressed(npz_filename, array_lungs=array_lungs)
    
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        
        print(message)
        continue
    
    print(
        'o',
        patientid, thres, box,
        array.shape, array_resampled.shape, array_lungs.shape
        )


# ----------------------
# patientid = '081f4a90f24ac33c14b61b97969b7f81'
# patientid = '5b642ed9150bac6cbd58a46ed8349afc' # cancer
# patientid = '174c5f7c33ca31443208ef873b9477e5' # unknown
# patientid = '3043b5cc9733054f3ab5399c3c810406' # not-cancer
# patientid = 'a6195eb9162fc40d5fa2fdb1239dd788' # not-cancer
# patientid = 'f82560aeea0309873716efe3aa71ef0a' # not-cancer
# patientid = '7842c108866fccf9b1b56dca68fc355e' # cancer
# patientid = 'a3cb12d3b8d5c64fa55f27208fe13a07' # not-cancer
# patientid = '0ca943d821204ceb089510f836a367fd' # not-cancer
# patientid = '0eafe9b9182b80c6d67015a2062f5143' # not-cancer
# patientid = '24031ba88c58797148475f6d4d5a085b' # not-cancer

# redux1 = np.prod(array_resampled.shape)/np.prod(array.shape)
# redux2 = np.prod(array_lungs.shape)/np.prod(array_resampled.shape)

# print('Original shape:', array.shape)
# print('Resample shape:', array_resampled.shape, 'Reduction:', redux1)
# print('Lungs box shape:', array_lungs.shape, 'Reduction:', redux2)
# print('Total reduction:', redux1*redux2)
# print(thres)