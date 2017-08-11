from keras import backend as K
from keras.layers import (Input, Conv3D, MaxPooling3D, UpSampling3D, Dense,
                          Activation, BatchNormalization, Dropout, Concatenate, Lambda)
from keras.models import Model

from kdsb17.layers import SpatialPyramidPooling3D
from kdsb17.losses import gmd_log_likelihood

# TODO: Implement predict method.
# TODO: Implement fit_generator method.
# TODO: Implement _build_callbacks method.
# TODO: Implement load_from_file method.


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
                 optimizer='adam', batch_normalization=False, dropout_rate=0.5):
        self.nb_filters_per_layer = nb_filters_per_layer
        self.kernel = (3, 3, 3)
        self.n_gaussians = n_gaussians
        self.optimizer = optimizer
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        self._cae3d = None
        self._classifier = None

    def _build_encoder_layers(self, layer):
        """Stacks sequence of conv/pool layers to make the encoder half.
        """

        for nb_filters in self.nb_filters_per_layer:
            layer = Conv3D(nb_filters, kernel=self.kernel, padding='same')(layer)
            if self.batch_normalization:
                layer = BatchNormalization(mode=0, axis=1)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling3D((2, 2, 2), padding='same')(layer)

        return layer

    def _build_decoder_layers(self, layer):
        """Stacks sequence of conv/up layers to make the decoder half.
        """

        for nb_filters in self.nb_filters_per_layer[::-1]:
            layer = Conv3D(nb_filters, kernel=self.kernel, padding='same')(layer)
            if self.batch_normalization:
                layer = BatchNormalization(mode=0, axis=1)(layer)
            layer = Activation('relu')(layer)
            layer = UpSampling3D((2, 2, 2))(layer)

        return layer

    def _build_gmd_layers(self, layer):
        """Builds the 3D CAE output layer that parametrizes a Gaussian Mixture Density.
        """

        prior = Conv3D(self.n_gaussians, kernel=self.kernel, activation='softmax', padding='same')(layer)

        mu = Conv3D(self.n_gaussians, kernel=self.kernel, padding='same')(layer)

        sigma = Conv3D(self.n_gaussians, kernel=self.kernel, padding='same')(layer)
        sigma = Lambda(K.exp)(sigma)

        gmd = Concatenate(axis=4)([prior, mu, sigma])

        return gmd

    def _build_classifier_layers(self, layer):
        """Builds layers for classification on top of encoder layers.
        """

        h = SpatialPyramidPooling3D((1, 2, 4))(layer)
        h = Dense(64, activation='sigmoid')(h)
        h = Dropout(self.dropout_rate)(h)
        h = Dense(32, activation='sigmoid')(h)
        h = Dropout(self.dropout_rate)(h)
        y = Dense(1, activation='sigmoid')(h)

        return y

    def build_models(self):
        """Builds the two models.
        """

        input_layer = Input(shape=(None, None, None, 1))

        # ---------- 3D Convolutional Autoencoder ----------
        encoded = self._build_encoder_layers(input_layer)
        decoded = self._build_decoder_layers(encoded)

        # Output layer parametrizes a Gaussian Mixture Density.
        y1 = self._build_gmd_layers(decoded)
        self._cae3d = Model(input_layer, y1)
        self._cae3d.compile(optimizer=self.optimizer, loss=gmd_log_likelihood)

        # ------------------- Classifier -------------------
        # Inherits encoder layers from 3D CAE, and performs binary classification.
        y2 = self._build_classifier_layers(encoded)
        self._classifier = Model(input_layer, y2)
        self._classifier.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def summary(self):
        print('Summary of 3D Convolutional Autoencoder:')
        self._cae3d.summary()
        print('Summary of Classifier:')
        self._classifier.summary()
