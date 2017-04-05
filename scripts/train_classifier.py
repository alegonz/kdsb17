import os
import sys
import time
sys.path.append('/data/code/')

import numpy as np
np.random.seed(1988)

from keras import backend
from keras.layers import Input, Dense, Convolution3D, MaxPooling3D, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from kdsb17.layers import SpatialPyramidPooling3D
from kdsb17.trainutils import Generator3dCNN
from kdsb17.fileutils import makedir

print('image_dim_ordering:', backend.image_dim_ordering())

# Data file parameters
models_path = '/data/models'
data_path = '/data/data'
dataset = 'npz_2mm_ks3_05p'
in_sample_csv_path = '/data/data/stage1_labels.csv'

# Training parameters
nb_epoch = 2
optimizer = 'adam'

# Define model
input_array = Input(shape=(1, None, None, None))
h = Convolution3D(64, 3, 3, 3, border_mode='valid')(input_array)
h = Activation(activation='relu')(h)
h = MaxPooling3D((2, 2, 2), border_mode='valid')(h)

h = Convolution3D(96, 3, 3, 3, border_mode='valid')(h)
h = Activation(activation='relu')(h)
h = MaxPooling3D((2, 2, 2), border_mode='valid')(h)

h = Convolution3D(96, 3, 3, 3, border_mode='valid')(h)
h = Activation(activation='relu')(h)
h = SpatialPyramidPooling3D([1, 2, 4])(h)

h = Dense(256, activation='sigmoid')(h)
h = Dense(256, activation='sigmoid')(h)
output_array = Dense(1, activation='sigmoid')(h)

model = Model(input_array, output_array)
model.compile(optimizer=optimizer, loss='binary_crossentropy')
model.summary()

# Create data generators
train_path = os.path.join(data_path, dataset, 'train')
# nb_train_samples = len(os.listdir(train_path))
nb_train_samples = 20

validation_path = os.path.join(data_path, dataset, 'validation')
# nb_val_samples = len(os.listdir(validation_path))
nb_val_samples = 20

train_gen_factory = Generator3dCNN(train_path, labels_path=in_sample_csv_path,
                                   mean=-350, rescale_map=((-1000, -1), (400, 1)),
                                   random_rotation=True, random_offset_range=(-60, 60))

val_gen_factory = Generator3dCNN(validation_path, labels_path=in_sample_csv_path,
                                 mean=-350, rescale_map=((-1000, -1), (400, 1)),
                                 random_rotation=False, random_offset_range=None)

train_generator = train_gen_factory.for_binary_classifier_chunked(chunk_size=100)
validation_generator = val_gen_factory.for_binary_classifier_chunked(chunk_size=100)

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
