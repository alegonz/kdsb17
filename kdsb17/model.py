import os

import numpy as np
from keras import backend as K
from keras.layers import (Input, Conv3D, Conv3DTranspose, Dense,
                          Activation, BatchNormalization, Dropout, Concatenate, Lambda)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from kdsb17.layers import SpatialPyramidPooling3D
from kdsb17.callbacks import BatchLossCSVLogger
from kdsb17.losses import gmd_log_likelihood

# TODO: Freeze encoding layers in classifier submodel. We may need to divide the submodel compilation in separate steps.


class LungNet(object):
    """Builds and compiles a set of two models for representation learning and classification of 3D CT lung scans:

    1) 3D Convolutional Autoencoder:
        Performs feature learning on CT scan patches. The network structure is as follows:

        Input -> Encoder -> Decoder -> Output

        The output parametrizes a Gaussian Mixture Density.

    2) Classifier:
        Classifies full CT lung scans, using the features learned by the 3D Convolutional Autoencoder.
        The network structure is as follows:

        Input -> Encoder -> SpatialPyramidPooling -> Classifier -> Output

        The encoder weights are transferred from the autoencoder, the classifier is a stack of dense layers,
        and the output parametrizes a Bernoulli distribution on the class labels.
    
    Args:
        nb_filters_per_layer (tuple): Number of filters (feature maps) per layer.
            The number of layers (pairs Conv/Pool and Conv/Up) is inferred from the length of the tuple.
        optimizer (str): Keras optimizer name.
        batch_normalization (bool): Use (True) batch normalization or not (false).
    """

    def __init__(self, nb_filters_per_layer=(96, 128, 128), n_gaussians=2,
                 batch_normalization=False, dropout_rate=0.5,
                 optimizer='adam', es_patience=10,
                 model_path='/tmp/', weights_name_format='weights.{epoch:02d}-{val_loss:.6f}.hdf5'):

        self.nb_filters_per_layer = nb_filters_per_layer
        self.kernel = (3, 3, 3)
        self.n_gaussians = n_gaussians

        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate

        self.optimizer = optimizer
        self.es_patience = es_patience

        self.model_path = model_path
        self.weights_name_format = weights_name_format

        self._input_layer = None
        self._output_layer = {'cae3d': None, 'classifier': None}
        self._submodels = {'cae3d': None, 'classifier': None}
        self._loss = {'cae3d': gmd_log_likelihood, 'classifier': 'binary_crossentropy'}
        self._built = {'cae3d': False, 'classifier': False}
        self._layers_built = False

    def _build_callbacks(self, subfolder=''):
        """Builds callbacks for training model.
        """

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_path, subfolder, self.weights_name_format),
                                       monitor='val_loss', save_best_only=True, save_weights_only=True)

        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.es_patience)

        epoch_logger = CSVLogger(os.path.join(self.model_path, subfolder, 'epoch_log.csv'))
        batch_logger = BatchLossCSVLogger(os.path.join(self.model_path, subfolder, 'batch_log.csv'))

        return [checkpointer, early_stopper, epoch_logger, batch_logger]

    def _build_encoder_layers(self, layer):
        """Stacks sequence of conv/pool layers to make the encoder half.
        """

        for i, nb_filters in enumerate(self.nb_filters_per_layer):

            layer = Conv3D(nb_filters, kernel_size=self.kernel, strides=(2, 2, 2),
                           padding='same', name=('encoder_conv_%d' % i))(layer)

            if self.batch_normalization:
                layer = BatchNormalization(mode=0, axis=1, name=('encoder_bn_%d' % i))(layer)

            layer = Activation('relu', name=('encoder_act_%d' % i))(layer)

        return layer

    def _build_decoder_layers(self, layer):
        """Stacks sequence of conv/up layers to make the decoder half.
        """

        for i, nb_filters in enumerate(self.nb_filters_per_layer[::-1]):

            layer = Conv3DTranspose(nb_filters, kernel_size=self.kernel, strides=(2, 2, 2),
                                    padding='same', name=('decoder_conv_%d' % i))(layer)

            if self.batch_normalization:
                layer = BatchNormalization(mode=0, axis=1, name=('decoder_bn_%d' % i))(layer)

            layer = Activation('relu', name=('decoder_act_%d' % i))(layer)

        return layer

    def _build_gmd_layers(self, layer):
        """Builds the 3D CAE output layer that parametrizes a Gaussian Mixture Density.
        """

        prior = Conv3D(self.n_gaussians, kernel_size=self.kernel, activation='softmax', padding='same', name='prior')(layer)

        mu = Conv3D(self.n_gaussians, kernel_size=self.kernel, padding='same', name='mu')(layer)

        sigma = Conv3D(self.n_gaussians, kernel_size=self.kernel, padding='same', name='sigma_raw')(layer)
        sigma = Lambda(K.exp, name='sigma')(sigma)

        gmd = Concatenate(axis=4, name='gmd')([prior, mu, sigma])

        return gmd

    def _build_classifier_layers(self, layer):
        """Builds layers for classification on top of encoder layers.
        """

        h = SpatialPyramidPooling3D((1, 2, 4), name='spp3d')(layer)
        h = Dense(64, activation='sigmoid')(h)
        h = Dropout(self.dropout_rate)(h)
        h = Dense(32, activation='sigmoid')(h)
        h = Dropout(self.dropout_rate)(h)
        y = Dense(1, activation='sigmoid')(h)

        return y

    def _build_layers(self):
        """Builds all layers
        """
        if self._layers_built:
            print('Layers are already built.')
            return

        self._input_layer = Input(shape=(None, None, None, 1))

        # 3D Convolutional Autoencoder
        encoded = self._build_encoder_layers(self._input_layer)
        decoded = self._build_decoder_layers(encoded)

        # Output layer parametrizes a Gaussian Mixture Density.
        self._output_layer['cae3d'] = self._build_gmd_layers(decoded)

        # Classifier
        # Inherits encoder layers from 3D CAE, and performs binary classification.
        self._output_layer['classifier'] = self._build_classifier_layers(encoded)

        self._layers_built = True

    def build_submodel(self, name, freeze=None):
        """Build the specified submodel.

        Args:
            name (str): Name of submodel.
            freeze (list): Name of layers to freeze in training.
        """
        if not self._layers_built:
            self._build_layers()

        self._submodels[name] = Model(self._input_layer, self._output_layer[name])

        # Freeze layers
        if freeze is not None:
            for layer_name in freeze:
                layer = self._submodels[name].get_layer(name=layer_name)
                layer.trainable = False

        self._submodels[name].compile(optimizer=self.optimizer, loss=self._loss[name])
        self._built[name] = True

    def summary(self):
        """Print summary of models to stdout.
        """
        for name, submodel in self._submodels.items():
            if self._built[name]:
                print('Summary of %s:' % name)
                submodel.summary()
            else:
                print('%s is not built yet.' % name)

    def load_weights_from_file(self, files):
        """Load the weights of the submodels from files.

        Args:
            files (dict): A dictionary with the paths to the weights of each submodel.
                The keys must match the name of the submodels.
        """
        for name, path in files.items():
            if name in self._submodels:
                if not self._built[name]:
                    raise ValueError('%s is not built. Build it first by calling build_submodel().' % name)
                self._submodels[name].load_weights(path, by_name=True)
                print('Loaded weights for %s.', name)

    def fit_submodel(self, name,
                     train_generator, steps_per_epoch, epochs,
                     validation_generator, validation_steps):
        """Fits a submodel on data yielded batch-by-batch by a generator.

        Args:
            name (str): Name of the submodel. Either 'cae3d' or 'classifier'.
            train_generator: A data generator that yields (x, y) tuples of training data/labels.
            steps_per_epoch: Steps (number of batches) per epoch.
            epochs: Number of epochs.
            validation_generator: A data generator that yields (x, y) tuples of validation data/labels.
            validation_steps: Validation steps (number of batches).

        Returns:
            Keras History object with history of training losses.
       """
        self._check_submodel_name(name)

        callbacks = self._build_callbacks(name)

        history = self._submodels[name].fit_generator(generator=train_generator,
                                                      steps_per_epoch=steps_per_epoch,
                                                      epochs=epochs,
                                                      callbacks=callbacks,
                                                      validation_data=validation_generator,
                                                      validation_steps=validation_steps)

        return history

    @staticmethod
    def _check_input_array(x):
        if x.dtype != 'float32':
            raise ValueError('Input array must be of type float32.')
        if x.ndim != 5:
            raise ValueError('Input array must have exactly 5 dimensions: (samples, z, y, z, channels)')
        if x.shape[-1] != 1:
            raise ValueError('Size of last dimension of input array must be exactly 1.')

    @staticmethod
    def _check_submodel_name(name):
        if name not in ('cae3d', 'classifier'):
            raise ValueError('Invalid submodel name. Must be either \'cae3d\' or \'classifier\'.')

    def predict(self, x, submodel_name):
        """Predict from submodel.

        Args:
            x (numpy.ndarray): Input array of shape (samples, z, y, x, 1) and type float32.
            submodel_name (str): Name of the submodel. Either 'cae3d' or 'classifier'.

        Returns:
            If submodel_name is 'cae3d':
                Array of the same shape and type as the input containing the 3D CAE reconstruction.
                Each voxel y of the reconstruction is predicted as:
                    y = mu[K] where K = argmax(k)(prior[k]/sigma[k]), k is the index of each gaussian in the mixture.

                    Ref: Bishop C. M., Mixture Density Networks, 1994.
                    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf

            If submodel_name is 'classifier':
                A scalar indicating the probability of cancer within one year.
        """
        self._check_input_array(x)
        self._check_submodel_name(submodel_name)

        submodel = self._submodels[submodel_name]

        pred = submodel.predict(x)

        if submodel_name == 'cae3d':
            prior, mu, sigma = np.split(pred, axis=4, indices_or_sections=3)

            which = (prior/sigma).argmax(axis=4)
            sample, z, y, x = np.indices(x.shape[:-1])

            return np.expand_dims(mu[sample, z, y, x, which], axis=4)

        elif submodel_name == 'classifier':
            return pred
