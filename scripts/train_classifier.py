import os
import sys
import time
sys.path.append('/data/code/')

import numpy as np
np.random.seed(1988)

from keras import backend
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from kdsb17.layers import SpatialPyramidPooling3D
from kdsb17.trainutils import Generator3dCNN, BatchLossCSVLogger
from kdsb17.fileutils import makedir

print('image_dim_ordering:', backend.image_dim_ordering())

# Data file parameters
models_path = '/data/models'
data_path = '/data/data/features/20170407_174348/weights.17-0.432386.hdf5'
dataset = 'npz_2mm_ks3_05p'
in_sample_csv_path = '/data/data/stage1_labels.csv'

# Training parameters
nb_epoch = 20
optimizer = 'adam'

# Define model
input_array = Input(shape=(128, None, None, None))
h = SpatialPyramidPooling3D([1, 2, 4])(input_array)
h = Dense(64, activation='sigmoid')(h)
h = Dropout(0.5)(h)
h = Dense(32, activation='sigmoid')(h)
h = Dropout(0.5)(h)
output_array = Dense(1, activation='sigmoid')(h)

model = Model(input_array, output_array)
model.compile(optimizer=optimizer, loss='binary_crossentropy')
model.summary()

# Create data generators
train_path = os.path.join(data_path, dataset, 'train')
datagen_train = Generator3dCNN(train_path, labels_path=in_sample_csv_path)
nb_train_samples = len(datagen_train.patients)

validation_path = os.path.join(data_path, dataset, 'validation')
datagen_val = Generator3dCNN(validation_path, labels_path=in_sample_csv_path)
nb_val_samples = len(datagen_val.patients)

train_generator = datagen_train.for_binary_classifier_full(array_type='cae3d_features')
val_generator = datagen_val.for_binary_classifier_full(array_type='cae3d_features')

# Create callbacks
time_string = time.strftime('%Y%m%d_%H%M%S')
out_path = makedir(os.path.join(models_path, 'classifier', time_string))  # dir for model files and log
weights_template = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

checkpointer = ModelCheckpoint(filepath=os.path.join(out_path, weights_template),
                               monitor='val_loss', save_best_only=True)

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

csv_logger = CSVLogger(os.path.join(out_path, 'epoch_log.csv'))
batch_logger = BatchLossCSVLogger(os.path.join(out_path, 'batch_log.csv'))

# Train model
model.fit_generator(generator=train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
                    validation_data=val_generator, nb_val_samples=nb_val_samples,
                    callbacks=[checkpointer, early_stopper, csv_logger, batch_logger])
