from keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model


class CAE3d:

    def __init__(self, nb_filters_per_layer=(96, 128, 128), optimizer='adam'):
        self.nb_filters_per_layer = nb_filters_per_layer
        self.optimizer = optimizer
        self.model = None

    def _encoder(self, layer):

        for nb_filters in self.nb_filters_per_layer:
            layer = Convolution3D(nb_filters, 3, 3, 3, activation='relu', border_mode='same')(layer)
            layer = MaxPooling3D((2, 2, 2), border_mode='same')(layer)

        return layer

    def _decoder(self, layer):

        for nb_filters in self.nb_filters_per_layer[::-1]:
            layer = Convolution3D(nb_filters, 3, 3, 3, activation='relu', border_mode='same')(layer)
            layer = UpSampling3D((2, 2, 2))(layer)

        return layer

    def compile(self):

        input_layer = Input(shape=(1, None, None, None))
        last_hidden_layer = self._decoder(self._encoder(input_layer))

        # Output layer has linear activation because the array is not an image.
        output_layer = Convolution3D(1, 3, 3, 3, border_mode='same')(last_hidden_layer)

        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer=self.optimizer, loss='mse')
