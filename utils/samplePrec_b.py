"""Update prec_b: inverse of regression coefficient variance."""
import numpy as np
import numpy.random as nr


def samplePrec_b(X, S, B, P, prec_ba, prec_bb, prec_b_iso):
    """Update precision B using Gamma(a, b) hyperprior."""
    if prec_b_iso:
        post_shape = prec_ba + 0.5 * np.sum(S)
        post_scale = 1. / (prec_bb + 0.5 * np.trace(np.dot(B.T, B)))
        prec_b = np.ones(P) * nr.gamma(post_shape, post_scale)

    else:
        prec_b = np.zeros(P)
        for p in range(P):
            Bp = B[:, [p]]
            post_shape = prec_ba + 0.5 * np.sum(S[:, p])
            post_scale = 1. / (prec_bb + 0.5 * np.dot(Bp.T, Bp))
            prec_b[p] = nr.gamma(post_shape, post_scale)
        assert(all(prec_b > 0))
    return prec_b


def samplePrec_bb(P, prec_b, prec_ba, prec_bb, prec_bb_a, prec_bb_b):
    """Update prec_bb to share power across the factors."""
    post_shape = prec_bb_a + prec_ba*P
    post_scale = 1. / (prec_bb_b + np.sum(prec_b))
    prec_bb = nr.gamma(post_shape, post_scale)

    return prec_bb
