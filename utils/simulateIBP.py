"""Simulate the Indian buffet process."""
import numpy as np
import numpy.random as nr


def simulateIBP(IBPa, D):
    """Initialize latent feature binary matrix Z."""
    Z = np.ones((0, 0))
    # Use while loop to prevent from generating K=0 feature assignment matrix
    while True:
        for i in range(1, D + 1):
            # Sample exsisting features
            # Z.sum(axis=0)/i: compute the popularity of every dish, computes
            # the probability of sampling that dish
            zi = (nr.uniform(0, 1, (1, Z.shape[1])) <
                  (Z.sum(axis=0).astype(float) / i))
            # Sample new features
            # Learning a value from the poisson distribution (IBPa/D)
            knew = nr.poisson(IBPa * 1.0 / i)
            zi = np.hstack((zi, np.ones((1, knew))))
            Z = np.hstack((Z, np.zeros((Z.shape[0], knew))))
            Z = np.vstack((Z, zi))
        if Z.shape[1] > 0:
            break
        else:
            Z = np.ones((0, 0))

    assert(Z.shape[0] == D)
    return Z
