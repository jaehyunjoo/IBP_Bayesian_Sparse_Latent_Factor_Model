"""Update IBP parameter alpha."""
import numpy as np
import numpy.random as nr


def sampleIBPa(IBPa_a, IBPa_b, K, D):
    """Update IBP alpha using Ga(a, b) hyperprior."""
    postshape = IBPa_a + K
    H_D = np.array([range(D)]) + 1.0
    H_D = np.sum(1.0 / H_D)
    postscale = 1.0 / (IBPa_b + H_D)
    IBPa = nr.gamma(postshape, scale=postscale)
    return IBPa
