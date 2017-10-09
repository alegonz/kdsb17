from keras import backend as K
from keras.engine import Layer


class ShiftedELU(Layer):
    """Shifted Exponential Linear Unit.
    It follows:
    `f(x) =  alpha * (exp(x) - 1.) + shift for x < 0`,
    `f(x) = x + shift for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: scale for the negative factor.
        shift: shift value for the factor(offset).
    """

    def __init__(self, shift=1.0, alpha=1.0, **kwargs):
        super(ShiftedELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.shift = K.cast_to_floatx(shift)
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.elu(inputs, self.alpha) + self.shift

    def get_config(self):
        config = {'alpha': float(self.alpha), 'shift': float(self.shift)}
        base_config = super(ShiftedELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
