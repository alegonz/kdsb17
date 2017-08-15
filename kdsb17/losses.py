from keras import backend as K

HALF_LOG_TWOPI = 0.91893853320467267  # (1/2)*log(2*pi)


def build_gmd_log_likelihood(input_shape, m):
    """Build log-likelihood loss for Gaussian Mixture Densities.

    Args:
        input_shape (tuple): Shape of input 3D array.
        m (int): Number of gaussians in the mixture.

    Returns:
        Loss function.
    """

    # TODO: Add check for integers in input_shape
    if len(input_shape) != 3 or any([d <= 0 for d in input_shape]):
        raise ValueError('input_shape must be 3D with positive dimensions.')
    if not (m > 0 and isinstance(m, int)):
        raise ValueError('n_gaussian must be a positive integer.')

    def _gmd_log_likelihood(y_true, y_pred):
        """Log-likelihood loss for Gaussian Mixture Densities.
        Currently only supports tensorflow backend.

        Args:
            y_true (tensor): A tensor of shape (samples, z*y*x) with the flattened true values.
            y_pred (tensor): Tensor of shape (samples, m*(z*y*x + 2)), where m is the number of gaussians.
                The second dimension encodes the following parameters (in that order):
                1) m log-priors (outputs of a log-softmax activation layer)
                2) m variances (outputs of a Shifted ShiftedELU activation layer)
                3) m*z*y*x means (outputs of a linear activation layer)

        Returns:
            Average negative log-likelihood of each sample.
        """
        z, y, x = input_shape
        splits = [m, m, m*z*y*x]

        # Get GMD parameters
        # Parameters are concatenated along the second axis
        # numpy.split expect sizes, not locations
        log_prior, sigma_sq, mu = K.tf.split(y_pred, num_or_size_splits=splits, axis=1)

        y_true = K.expand_dims(y_true, axis=2)
        mu = K.reshape(mu, [-1, z*y*x, m])  # -1 is for the sample dimension
        dist = K.sum(K.square(y_true - mu), axis=1)

        exponent = log_prior - m*HALF_LOG_TWOPI - (m/2.0)*K.log(sigma_sq) - (1/2.0)*dist/sigma_sq

        return -K.logsumexp(exponent, axis=1)

    return _gmd_log_likelihood
