from keras import backend as K


def log_softmax(x, axis=-1):
    """Log-softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the log-softmax normalization is applied.
    # Returns
        Tensor, output of log-softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.tf.nn.log_softmax(x, dim=axis)
    elif ndim > 2:
        e = x - K.max(x, axis=axis, keepdims=True)
        s = K.log(K.sum(e, axis=axis, keepdims=True))
        return e - s
    else:
        raise ValueError('Cannot apply log-softmax to a tensor that is 1D')
