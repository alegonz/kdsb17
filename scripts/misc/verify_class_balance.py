import os
import sys
from glob import glob

sys.path.append('/data/code/')

from kdsb17.trainutils import read_labels

data_path = '/data/data'
dataset = 'npz_2mm_ks3_05p'
subset = 'all'

# Read labels
in_sample_csv_path = os.path.join(data_path, 'stage1_labels.csv')
in_sample_labels = read_labels(in_sample_csv_path, header=True)

# Get list of patients in subset
paths = glob(os.path.join(data_path, dataset, subset, '*.npz'))
patient_list = [os.path.basename(path).split('.')[0] for path in paths]

# Calculate class balance
n0 = sum([1 if label == 0 else 0
          for patient, label in in_sample_labels.items() if patient in patient_list])

n1 = sum([1 if label == 1 else 0
          for patient, label in in_sample_labels.items() if patient in patient_list])

print('Total samples:', n0 + n1)
print('Cancer:', n1, '(%.2f)' % (n1/(n0 + n1)))
print('Cancer:', n0, '(%.2f)' % (n0/(n0 + n1)))
