from itertools import product
from keras.engine.topology import Layer
from keras import backend as K


class SpatialPyramidPooling3D(Layer):
    """Three-dimensional Spatial Pyramid Pooling.

    An extension of the method of He. K et al (https://arxiv.org/abs/1406.4729) to the 3D case.
    Currently, there is no overlap between bins, i.e. stride size = bin size

    Args:
        nb_bins_per_level (list of int): Number of bins into which the axes of the input volume is partitioned
            for pooling (per channel). Each element of the list represents a pooling level in the pyramid.

            e.g. if [1, 2, 4] then each volume (each channel) is divided and pooled across
            1 (=1**3) + 8 (=2**3) + 64 (=4**3) = 73 regions in 3 levels, yielding 73*nb_channels features.

        pooling_mode (str): Either 'max' for max pooling or 'mean' for average pooling.

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

    def __init__(self, nb_bins_per_level, pooling_mode='max', **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        self.nb_bins_per_level = nb_bins_per_level

        if pooling_mode == 'max':
            self.poolfunc = K.max
        elif pooling_mode == 'mean':
            self.poolfunc = K.mean
        else:
            raise ValueError('Invalid pooling mode; must be either max or mean.')

        super(SpatialPyramidPooling3D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[4]

        super(SpatialPyramidPooling3D, self).build(input_shape)

    def get_output_shape_for(self, input_shape):

        nb_samples = input_shape[0]
        nb_outputs_per_channel = sum([nb_bins ** 3 for nb_bins in self.nb_bins_per_level])
        nb_features = self.nb_channels * nb_outputs_per_channel

        return nb_samples, nb_features

    def call(self, x, mask=None):

        if self.dim_ordering == 'th':
            __, __, len_i, len_j, len_k = x.shape
        elif self.dim_ordering == 'tf':
            __, len_i, len_j, len_k, __ = x.shape

        outputs = []

        for nb_bins in self.nb_bins_per_level:
            bin_size_i, bin_size_j, bin_size_k = (len_i // nb_bins, len_j // nb_bins, len_k // nb_bins)

            for i, j, k in product(range(nb_bins), range(nb_bins), range(nb_bins)):
                # each combination of i,j,k is a unique box upon which pooling is performed
                i1, i2 = i * bin_size_i, (i + 1) * bin_size_i
                j1, j2 = j * bin_size_j, (j + 1) * bin_size_j
                k1, k2 = k * bin_size_k, (k + 1) * bin_size_k

                if self.dim_ordering == 'th':
                    pooled_features = self.poolfunc(x[:, :, i1:i2, j1:j2, k1:k2], axis=(2, 3, 4))

                elif self.dim_ordering == 'tf':
                    pooled_features = self.poolfunc(x[:, i1:i2, j1:j2, k1:k2, :], axis=(1, 2, 3))

                outputs.append(pooled_features)

        return K.concatenate(outputs, axis=1)
