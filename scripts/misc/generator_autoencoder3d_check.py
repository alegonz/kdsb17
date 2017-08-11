import os
import sys

import numpy as np

np.random.seed(1988)

sys.path.append('/data/code/')
from kdsb17.utils.datagen import Generator3dCNN

data_path = '/data/data'
dataset = 'npz_2mm_ks3_05p'
in_sample_csv_path = '/data/data/stage1_labels.csv'
train_path = os.path.join(data_path, dataset, 'train')

train_gen_factory = Generator3dCNN(train_path, labels_path=in_sample_csv_path,
                                   random_rotation=False, random_offset_range=None)

generator = train_gen_factory.for_autoencoder_chunked(input_size=(32, 32, 32), batch_size=128, chunk_size=100)

for i, (x1, x2) in enumerate(generator):
    print(i, x1.shape, x1.min(), x1.max(), x1.mean())
    print(i, x2.shape, x2.min(), x2.max(), x2.mean())
    # show_slices(x1[0, 0], filename=os.path.join('/data/analysis/temp', str(i)+'.png'), every=5, cols=10)
    if i >= 20:
        break
