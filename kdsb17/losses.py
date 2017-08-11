import numpy as np
from keras import backend as K


def _tf_gaussian(y, mu, sigma):
    """Calculates probability density from a Gaussian PDF.
    Args:
        y (tensor): Value at which the PDF is evaluated.
        mu (tensor): Mean.
        sigma (tensor): Standard deviation.

    Returns:
        Probability density at specified value.
    """
    c = 1.0 / np.sqrt(2 * np.pi)
    exponent = -(((y - mu) / sigma) ** 2) / 2.0
    return c * (1.0 / sigma) * K.exp(exponent)


def gmd_log_likelihood(y_true, y_pred):
    """Log-likelihood loss for Gaussian Mixture Densities.
    Currently only supports tensorflow format.

    Args:
        y_true (tensor): A tensor of shape (samples, z, y, x, 1) with the true values.
        y_pred (tensor): Tensor of shape (samples, z, y, x, K*3), where K is the number of gaussians.
            The second dimension encodes the K priors, K means and K standard deviations of each gaussian.

    Returns:
        Average negative log-likelihood across samples.
    """
    # Get GMD parameters
    # Assume parameters are concatenated along the last axis
    priors, mu, sigma = K.tf.split(y_pred, num_or_size_splits=3, axis=4)

    # Compute negative log-likelihood
    z = _tf_gaussian(y_true, mu, sigma) * priors
    z = K.sum(z, axis=4, keep_dims=True)
    z = -K.log(z)

    return K.mean(z)
