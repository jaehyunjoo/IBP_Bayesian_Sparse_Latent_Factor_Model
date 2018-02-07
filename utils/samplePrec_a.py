"""Update prec_a: inverse of factor loading variance."""
import numpy as np
import numpy.random as nr


def samplePrec_a(X, Z, A, K, prec_aa, prec_ab, prec_a_iso):
    """Update precision A using Gamma(a, b) hyperprior."""
    if prec_a_iso:
        post_shape = prec_aa + 0.5 * np.sum(Z)
        post_scale = 1. / (prec_ab + 0.5 * np.trace(np.dot(A.T, A)))
        prec_a = np.ones(K) * nr.gamma(post_shape, post_scale)

    else:
        prec_a = np.zeros(K)
        for k in range(K):
            Ak = A[:, [k]]
            post_shape = prec_aa + 0.5 * np.sum(Z[:, k])
            post_scale = 1. / (prec_ab + 0.5 * np.dot(Ak.T, Ak))
            prec_a[k] = nr.gamma(post_shape, post_scale)
        assert(all(prec_a > 0))
    return prec_a


def samplePrec_ab(K, prec_a, prec_aa, prec_ab, prec_ab_a, prec_ab_b):
    """Update prec_ab to share power across the factors."""
    post_shape = prec_ab_a + prec_aa*K
    post_scale = 1. / (prec_ab_b + np.sum(prec_a))
    prec_ab = nr.gamma(post_shape, post_scale)

    return prec_ab
