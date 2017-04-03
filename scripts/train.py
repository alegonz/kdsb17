import os
import sys
import time
sys.path.append('/data/code/')

import numpy as np
np.random.seed(1969)

from keras.layers import Input, Dense, Dropout, Convolution3D, MaxPooling3D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from kdsb17.layers import SpatialPyramidPooling3D
from kdsb17.trainutils import create_generator
from kdsb17.fileutils import makedir

# Data file parameters
models_path = '/data/models'
data_path = '/data/data'
dataset = 'npz_2mm_ks3_05p'
in_sample_csv_path = '/data/data/stage1_labels.csv'

# Training paremeters
nb_epoch = 5

# Define model
input_array = Input(shape=(1, None, None, None))
h = Convolution3D(32, 3, 3, 3, border_mode='valid')(input_array)
h = MaxPooling3D((2, 2, 2), border_mode='valid')(h)
h = Convolution3D(64, 3, 3, 3, border_mode='valid')(h)
h = MaxPooling3D((2, 2, 2), border_mode='valid')(h)
h = Convolution3D(128, 3, 3, 3, border_mode='valid')(h)
h = SpatialPyramidPooling3D([1, 2, 4])(h)
h = Dense(1024, activation='sigmoid')(h)
h = Dropout(0.5)(h)
h = Dense(1024, activation='sigmoid')(h)
h = Dropout(0.5)(h)
output_array = Dense(1, activation='sigmoid')(h)

model = Model(input_array, output_array)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

# Create data generators
train_path = os.path.join(data_path, dataset, 'train')
nb_train_samples = len(os.listdir(train_path))

validation_path = os.path.join(data_path, dataset, 'validation')
nb_val_samples = len(os.listdir(validation_path))

train_generator = create_generator(train_path, labels_path=in_sample_csv_path,
                                   mean=-350, rescale_map=((-1000, -1), (400, 1)),
                                   rotate_randomly=True, random_offset_range=(-60, 60))

validation_generator = create_generator(validation_path, labels_path=in_sample_csv_path,
                                        mean=-350, rescale_map=((-1000, -1), (400, 1)),
                                        rotate_randomly=True, random_offset_range=(-60, 60))

# Create callbacks
time_string = time.strftime('%Y%m%d_%H%M%S')
out_path = makedir(os.path.join(models_path, time_string))  # output directory for model files and training log
weights_template = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

checkpointer = ModelCheckpoint(filepath=os.path.join(out_path, weights_template),
                               monitor='val_loss', save_best_only=True)

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

csv_logger = CSVLogger(os.path.join(out_path, 'training_log.csv'))

# Train model
model.fit_generator(generator=train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
                    validation_data=validation_generator, nb_val_samples=nb_val_samples,
                    callbacks=[checkpointer, early_stopper, csv_logger])
