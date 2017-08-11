import os
import sys

import numpy as np

np.random.seed(1988)

sys.path.append('/data/code/')
from kdsb17.utils.datagen import Generator3dCNN
from kdsb17.utils.plot import show_slices

data_path = '/data/data'
dataset = 'npz_2mm_ks3_05p'
in_sample_csv_path = '/data/data/stage1_labels.csv'
train_path = os.path.join(data_path, dataset, 'train')

train_gen_factory = Generator3dCNN(train_path, labels_path=in_sample_csv_path,
                                   mean=-350, rescale_map=((-1000, -1), (400, 1)),
                                   random_rotation=False, random_offset_range=None)
generator = train_gen_factory.for_binary_classifier()

for i, (x, y) in enumerate(generator):
    print(i, x.shape, x.min(), x.max(), x.mean(), y.shape, y)
    show_slices(x[0, 0], filename=os.path.join('/data/analysis/temp', str(i)+'.png'), every=5, cols=10)
    if i >= 20:
        break
