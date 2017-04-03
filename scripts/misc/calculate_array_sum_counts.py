import os
from glob import glob
import numpy as np

data_path = '/data/data/'
dataset = 'npz_2mm_ks3_05p'
subset = 'train'

paths = glob(os.path.join(data_path, dataset, subset, '*.npz'))

for path in paths:
    patient_id = os.path.basename(path).split('.')[0]

    with np.load(path) as data:
        x = data['array_lungs']

    s = x.sum(dtype='float32')
    n = np.prod(x.shape)

    print(patient_id, s, n)
