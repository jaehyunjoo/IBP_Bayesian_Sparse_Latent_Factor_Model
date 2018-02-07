"""Calculate Log likelihood."""
import numpy as np


def logLik(X, F, A, prec_x, N, D, prec_x_iso, H, B):
    """Likelihood based on model: X ~ N(FA, np.diag(1./prec_x))."""
    assert(F.shape[1] == N and A.shape[0] == D)

    lp = -0.5 * N * D * np.log(2*np.pi)
    lp += -0.5 * N * np.prod(1./prec_x)
    if prec_x_iso:
        E = X - np.dot(A, F)
        if H is not None:
            E -= np.dot(B, H)
        lp += -0.5 * prec_x[1] * np.trace(np.dot(E.T, E))
    else:
        for i in range(N):
            E_vec = X[:, [i]] - np.dot(A, F[:, [i]])
            if H is not None:
                E_vec -= np.dot(B, H[:, [i]])
            lp += -0.5 * np.dot(E_vec.T, np.dot(np.diag(prec_x), E_vec))

    return lp
