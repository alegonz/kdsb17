from keras import backend as K

HALF_LOG_TWOPI = 0.91893853320467267  # (1/2)*log(2*pi)


def build_gmd_log_likelihood(c, m):
    """Build log-likelihood loss for Gaussian Mixture Densities.

    Args:
        c (int): Number of output dimensions.
        m (int): Number of gaussians in the mixture.

    Returns:
        Loss function.
    """

    if not (c > 0 and isinstance(c, int)):
        raise ValueError('c must be a positive integer.')
    if not (m > 0 and isinstance(m, int)):
        raise ValueError('m must be a positive integer.')

    def _gmd_log_likelihood(y_true, y_pred):
        """Log-likelihood loss for Gaussian Mixture Densities.
        Currently only supports tensorflow backend.

        Args:
            y_true (tensor): A tensor of shape (samples, c) with the target values.
            y_pred (tensor): Tensor of shape (samples, m*(c + 2)), where m is the number of gaussians.
                The second dimension encodes the following parameters (in that order):
                1) m log-priors (outputs of a log-softmax activation layer)
                2) m variances (outputs of a ShiftedELU activation layer)
                3) m*c means (outputs of a linear activation layer)

        Returns:
            Average negative log-likelihood of each sample.
        """
        splits = [m, m, m*c]

        # Get GMD parameters
        # Parameters are concatenated along the second axis
        # tf.split expect sizes, not locations
        log_prior, sigma_sq, mu = K.tf.split(y_pred, num_or_size_splits=splits, axis=1)

        y_true = K.expand_dims(y_true, axis=2)
        mu = K.reshape(mu, [-1, c, m])  # -1 is for the sample dimension
        dist = K.sum(K.square(y_true - mu), axis=1)

        exponent = log_prior - c*HALF_LOG_TWOPI - (c/2.0)*K.log(sigma_sq) - (1/2.0)*dist/sigma_sq

        return -K.logsumexp(exponent, axis=1)

    return _gmd_log_likelihood
