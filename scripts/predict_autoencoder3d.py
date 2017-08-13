import os
import sys
sys.path.append('/data/code/')

import numpy as np

from keras.models import Model, load_model
from kdsb17.utils.datagen import GeneratorFactory
from kdsb17.utils.file import makedir


def predict_autoencoder3d(model_name=None, dataset=None):
    """
    Args:
        model_name (None or str): Path to model file relative to models folder.
            e.g. '20170407_174348/weights.17-0.432386.hdf5'
        dataset (None or str): Path to dataset folder relative to data folder.
            e.g. 'npz_2mm_ks3_05p/train'
    """

    if (model_name is None) and (dataset is None):
        try:
            _, model_name, dataset = sys.argv
        except ValueError:
            print('This script takes exactly two arguments, the model weights path and the dataset path.')

    print('Using model:', model_name)
    print('Using dataset:', dataset)

    data_path = '/data/data/'
    model_path = '/data/models/autoencoder3d'
    layer_name = 'convolution3d_3'
    features_path = '/data/data/features'

    out_path = makedir(os.path.join(features_path, model_name, dataset))

    # Load 3d convolutional autoencoder model and make intermediate model
    cae3d_model = load_model(os.path.join(model_path, model_name))

    intermediate_model = Model(input=cae3d_model.input,
                               output=cae3d_model.get_layer(layer_name).output)

    # Make generator for prediction
    data_gen = GeneratorFactory(data_path=os.path.join(data_path, dataset), labels_path=None)
    generator = data_gen.for_prediction(array_type='array_lungs')

    for patient_id, x, _ in generator:
        print('Extracting features of patient', patient_id)
        cae3d_features = intermediate_model.predict(x)
        npz_filename = os.path.join(out_path, patient_id + '.npz')
        np.savez_compressed(npz_filename, cae3d_features=cae3d_features)


if __name__ == '__main__':
    sys.exit(predict_autoencoder3d())
