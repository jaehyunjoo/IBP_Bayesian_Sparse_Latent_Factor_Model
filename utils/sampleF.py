"""Update factor score matrix F."""
import numpy as np
import numpy.random as nr


def sampleF(X, A, prec_x, N, D, K):
    """Sample new factor score based on normal prior."""
    psi_inv = np.diag(prec_x)
    prec = np.dot(A.T, np.dot(psi_inv, A)) + np.eye(K)
    prec_inv = np.linalg.inv(prec)
    Mu = np.dot(prec_inv, np.dot(A.T, np.dot(psi_inv, X)))
    L = np.linalg.cholesky(prec_inv)

    F = Mu + np.dot(L, nr.standard_normal((K, N)))

    return F
