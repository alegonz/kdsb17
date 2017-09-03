from itertools import product
from keras.engine.topology import Layer
from keras import backend as K


class SpatialPyramidPooling3D(Layer):
    """Three-dimensional Spatial Pyramid Pooling.

    An extension of the method of He. K et al (https://arxiv.org/abs/1406.4729) to the 3D case.
    Currently, there is no overlap between bins, i.e. stride size = bin size

    Args:
        nb_bins_per_level (tuple of int): Number of bins into which the axes of the input volume is partitioned
            for pooling (per channel). Each element of the list represents a pooling level in the pyramid.

            e.g. if [1, 2, 4] then each volume (each channel) is divided and pooled across
            1 (=1**3) + 8 (=2**3) + 64 (=4**3) = 73 regions in 3 levels, yielding 73*nb_channels features.

    # Input shape
        5D tensor with shape:
        `(samples, channels, dim1, dim2, dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, dim1, dim2, dim3, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, nb_features)` where nb_features is equal to
        nb_channels * sum([nb_bins**3 for nb_bins in nb_bins_per_layer])
    """

    def __init__(self, nb_bins_per_level, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        self.nb_bins_per_level = nb_bins_per_level
        self.nb_channels = None

        super(SpatialPyramidPooling3D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[4]

        super(SpatialPyramidPooling3D, self).build(input_shape)

    def get_config(self):
        config = {'nb_bins_per_level': self.nb_bins_per_level}
        base_config = super(SpatialPyramidPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        nb_samples = input_shape[0]
        nb_outputs_per_channel = sum([nb_bins ** 3 for nb_bins in self.nb_bins_per_level])
        nb_features = self.nb_channels * nb_outputs_per_channel

        return nb_samples, nb_features

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'th':
            len_i, len_j, len_k = input_shape[2], input_shape[3], input_shape[4]
        elif self.dim_ordering == 'tf':
            len_i, len_j, len_k = input_shape[1], input_shape[2], input_shape[3]

        outputs = []

        for nb_bins in self.nb_bins_per_level:
            bin_size_i = K.cast(len_i, 'int32') // nb_bins
            bin_size_j = K.cast(len_j, 'int32') // nb_bins
            bin_size_k = K.cast(len_k, 'int32') // nb_bins

            for i, j, k in product(range(nb_bins), range(nb_bins), range(nb_bins)):
                # each combination of i,j,k is a unique box upon which pooling is performed
                # the * operand works if Dimension*int, but not if int*Dimension. __rmul__ is not implemented.
                i1, i2 = bin_size_i * i, bin_size_i * (i + 1)
                j1, j2 = bin_size_j * j, bin_size_j * (j + 1)
                k1, k2 = bin_size_k * k, bin_size_k * (k + 1)

                if self.dim_ordering == 'th':
                    pooled_features = K.max(x[:, :, i1:i2, j1:j2, k1:k2], axis=(2, 3, 4))

                elif self.dim_ordering == 'tf':
                    pooled_features = K.max(x[:, i1:i2, j1:j2, k1:k2, :], axis=(1, 2, 3))

                outputs.append(pooled_features)

        return K.concatenate(outputs, axis=1)
