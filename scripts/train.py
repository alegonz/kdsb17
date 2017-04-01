import sys
sys.path.append('/data/code/')

import numpy as np
np.random.seed(1969)

from keras.layers import Input, Dense, Convolution3D, MaxPooling3D
from keras.models import Model

from kdsb17.layers import SpatialPyramidPooling3D
from kdsb17.trainutils import build_generator

train_path = '/data/data/train'
validation_path = '/data/data/validation'
in_sample_csv_path = '/data/data/stage1_labels.csv'
nb_val_samples = 280

# --- Define model
input_array = Input(shape=(1, None, None, None))
h = Convolution3D(4, 3, 3, 3, border_mode='valid')(input_array)
h = MaxPooling3D((2, 2, 2), border_mode='valid')(h)
h = Convolution3D(4, 3, 3, 3, border_mode='valid')(h)
h = MaxPooling3D((2, 2, 2), border_mode='valid')(h)
h = SpatialPyramidPooling3D([1, 2, 4])(h)
h = Dense(1024, activation='sigmoid')(h)
output_array = Dense(1, activation='sigmoid')(h)

model = Model(input_array, output_array)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

train_generator = build_generator(train_path, labels_path=in_sample_csv_path,
                                  mean=0, rescale_range=(-1000, 400), seed=2017)

validation_generator = build_generator(validation_path, labels_path=in_sample_csv_path,
                                       mean=0, rescale_range=(-1000, 400), seed=2017)

model.fit_generator(generator=train_generator, samples_per_epoch=50, nb_epoch=1,
                    validation_data=validation_generator, nb_val_samples=nb_val_samples)
