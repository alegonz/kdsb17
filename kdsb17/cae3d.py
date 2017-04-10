from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization
from keras.models import Model


class CAE3d:
    """Builds and compiles a 3D convolutional autoencoder.
    
    Args:
        nb_filters_per_layer (tuple): Number of filters (feature maps) per layer.
            The number of layers (pairs Conv/Pool and Conv/Up) is inferred from the length of the tuple.
        optimizer (str): Keras optimizer name.
        batch_normalization (bool): Use (True) batch normalization or not (false).
    """

    def __init__(self, nb_filters_per_layer=(96, 128, 128), optimizer='adam', batch_normalization=False):
        self.nb_filters_per_layer = nb_filters_per_layer
        self.optimizer = optimizer
        self.batch_normalization = batch_normalization
        self.model = None

    def _encoder(self, layer):
        """Stacks sequence of conv/pool layers to make the encoder half.
        
        Args:
            layer (keras.layer): layer upon which the encoder layers is stacked.
        Returns:
            Encoder layers.
        """

        for nb_filters in self.nb_filters_per_layer:
            layer = Convolution3D(nb_filters, 3, 3, 3, border_mode='same')(layer)
            if self.batch_normalization:
                layer = BatchNormalization(mode=0, axis=1)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling3D((2, 2, 2), border_mode='valid')(layer)

        return layer

    def _decoder(self, layer):
        """Stacks sequence of conv/up layers to make the decoder half.

        Args:
            layer (keras.layer): layer upon which the decoder layers is stacked.
        Returns:
            Decoder layers.
        """

        for nb_filters in self.nb_filters_per_layer[::-1]:
            layer = Convolution3D(nb_filters, 3, 3, 3, border_mode='same')(layer)
            if self.batch_normalization:
                layer = BatchNormalization(mode=0, axis=1)(layer)
            layer = Activation('relu')(layer)
            layer = UpSampling3D((2, 2, 2))(layer)

        return layer

    def compile(self):
        """Defines input, encoding, decoding and output layers, and compiles the model.
        """

        input_layer = Input(shape=(1, None, None, None))
        last_hidden_layer = self._decoder(self._encoder(input_layer))

        # Output layer has linear activation because the array is not an image.
        output_layer = Convolution3D(1, 3, 3, 3, border_mode='same')(last_hidden_layer)

        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer=self.optimizer, loss='mse')
