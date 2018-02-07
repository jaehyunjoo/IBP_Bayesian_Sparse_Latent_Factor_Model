"""Update prec_x: inverse of residual variance."""
import numpy as np
import numpy.random as nr


def samplePrec_x(E, N, D, prec_xa, prec_xb, prec_x_iso):
    """Update precision X using Ga(a, b) hyperprior."""
    if prec_x_iso:
        post_shape = prec_xa + N * D * 0.5
        post_scale = 1. / (prec_xb + 0.5 * np.trace(np.dot(E, E.T)))
        prec_x = np.ones(D) * nr.gamma(post_shape, post_scale)
    else:
        prec_x = np.zeros(D)
        for d in range(D):
            Ed = E[[d], :]
            post_shape = prec_xa + N * 0.5
            post_scale = 1. / (prec_xb + 0.5 * np.dot(Ed, Ed.T))

            prec_x[d] = nr.gamma(post_shape, post_scale)
            while True:
                if (1./prec_x[d] > 10e-10):
                    break
                else:
                    print("Psi %d lower than LOD; make it double." % (d+1))
                    prec_x[d] = prec_x[d]/2.
        assert(all(prec_x > 0))

    return prec_x


def samplePrec_xb(D, prec_x, prec_xa, prec_xb, prec_xb_a, prec_xb_b):
    """Update prec_xb to share power across the dimensions."""
    post_shape = prec_xb_a + prec_xa*D
    post_scale = 1. / (prec_xb_b + np.sum(prec_x))
    prec_xb = nr.gamma(post_shape, post_scale)

    return prec_xb
