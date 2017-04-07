import os
import sys
import time
sys.path.append('/data/code/')

import numpy as np
np.random.seed(1702)

from keras import backend
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from kdsb17.cae3d import CAE3d
from kdsb17.trainutils import Generator3dCNN, BatchLossCSVLogger
from kdsb17.fileutils import makedir

print('image_dim_ordering:', backend.image_dim_ordering())

# Data file parameters
models_path = '/data/models'
data_path = '/data/data'
dataset = 'npz_2mm_ks3_05p'
in_sample_csv_path = '/data/data/stage1_labels.csv'

# Training parameters
input_size = (16, 16, 16)  # (48, 48, 48)
nb_epoch = 20
batch_size = 96
chunk_size = 100
optimizer = 'adam'

# Define model
cae3d = CAE3d(nb_filters_per_layer=(64, 96, 128), optimizer=optimizer, batch_normalization=True)
cae3d.compile()
cae3d.model.summary()

# Create data generators
train_path = os.path.join(data_path, dataset, 'train')
# nb_train_samples = (490930//batch_size) * batch_size  # Closest multiple of batch size to 490,930 for size 32
# nb_train_samples = (149538//batch_size) * batch_size  # Closest multiple of batch size to 149,538 for size 48
nb_train_samples = 50000

validation_path = os.path.join(data_path, dataset, 'validation')
# nb_val_samples = (122952//batch_size) * batch_size  # Closest multiple of batch size to 122,952
# nb_val_samples = (37385//batch_size) * batch_size  # Closest multiple of batch size to 37,385
nb_val_samples = 12000

train_gen_factory = Generator3dCNN(train_path, labels_path=in_sample_csv_path,
                                   random_rotation=True, random_offset_range=None)

val_gen_factory = Generator3dCNN(validation_path, labels_path=in_sample_csv_path,
                                 random_rotation=False, random_offset_range=None)

train_generator = train_gen_factory.for_autoencoder_chunked(input_size=input_size,
                                                            batch_size=batch_size, chunk_size=chunk_size)

validation_generator = val_gen_factory.for_autoencoder_chunked(input_size=input_size,
                                                               batch_size=batch_size, chunk_size=chunk_size)

# Create callbacks
time_string = time.strftime('%Y%m%d_%H%M%S')
out_path = makedir(os.path.join(models_path, 'autoencoder3d', time_string))  # dir for model files and log
weights_template = 'weights.{epoch:02d}-{val_loss:.6f}.hdf5'

checkpointer = ModelCheckpoint(filepath=os.path.join(out_path, weights_template),
                               monitor='val_loss', save_best_only=True)

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

csv_logger = CSVLogger(os.path.join(out_path, 'epoch_log.csv'))
batch_logger = BatchLossCSVLogger(os.path.join(out_path, 'batch_log.csv'))

# Train model
cae3d.model.fit_generator(generator=train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
                          validation_data=validation_generator, nb_val_samples=nb_val_samples,
                          callbacks=[checkpointer, early_stopper, csv_logger, batch_logger],
                          nb_worker=1, max_q_size=1)
