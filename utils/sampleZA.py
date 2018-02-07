"""Update loading component A_dk."""
import numpy as np
import numpy.random as nr
# from sampleK import sampleK


def sampleZA(X, F, Z, A, prec_x, prec_a, IBPa, K, N, D, proposeK,
             prec_aa, prec_ab, prec_a_iso):
    """Update A_dk from Knowles and Ghahramani (2011)."""
    # Check Matrix type: Z - int, A - float
    assert(np.issubsctype(Z, np.int) and np.issubsctype(A, np.float))

    # Calculate vector F_k * F_k.T
    FF = np.dot(F, F.T).diagonal()
    E = X - np.dot(A, F)

    for d in range(D):
        for k in range(K):
            m_k = np.sum(Z[:, k]) - Z[d, k]

            # We add IBPa/k before loop for the fixed K case,
            # since it never proceeds for the singleton feature m_k=0.
            # if proposeK is False:
            #     m_k += IBPa/K

            if m_k > 0 or proposeK is False:
                # Evaluating error assuming A_dk = 0
                # Since E mat already subtracted an error attributed to A_dk,
                # we add it back here
                if Z[d, k]:
                    ek = E[[d], :] + A[d, k]*F[[k], :]
                else:
                    ek = E[[d], :]

                # Calculate Lambda = prec_x[d]*F_k*F_k.T + prec_a[k]
                Lambda = prec_x[d] * FF[k] + prec_a[k]
                Mu = (prec_x[d] * np.dot(F[[k], :], ek.T)) / Lambda

                # Calculate log prior ratio P(Z_dk=1|-) / P(Z_dk=0|-)
                if proposeK:
                    log_prior_ratio = np.log(m_k) - np.log(D-m_k)
                else:
                    log_prior_ratio = np.log(m_k+IBPa/K) - np.log(D-m_k)

                # Calculate log likelihood ratio P(X|Z_dk=1,-) / P(X|Z_dk=0,-)
                log_ll_ratio = 0.5*(np.log(prec_a[k])-np.log(Lambda))
                log_ll_ratio += 0.5*(Lambda*(Mu**2))

                # Calculate probability ratio
                log_prob_ratio = log_prior_ratio + log_ll_ratio
                prob_ratio = np.exp(-log_prob_ratio)

                # Calculate probability
                p1 = 1./(1 + prob_ratio)
                assert(not np.isnan(p1).any())

                # Update Z_dk and A_dk
                Z[d, k] = 1 if nr.uniform() < p1 else 0
                if Z[d, k]:
                    sd = np.sqrt(1./Lambda)
                    A[d, k] = nr.normal(Mu, sd)
                else:
                    A[d, k] = 0

                # Update a error term based on the change in A[d, k]
                # initially, ek was evaluated assuming A_dk = 0

                E[d, :] = ek - A[d, k] * F[k, :]
                ed2 = X[d, :] - np.dot(A[d, :], F)
                assert((E[d, :] - ed2 < 10e-8).all())

        # not necessary but maybe good to refresh
        # E[d, :] = X[d, :] - np.dot(A[d, :], F)

        if proposeK:

            poiMu = IBPa/float(D)
            Knew = nr.poisson(poiMu)
            assert(isinstance(Knew, int))

            (Z, A, F, FF, prec_a, E, K) = \
                sampleK(d, poiMu, Knew, N, D, K, X, F, Z, A, E, FF,
                        prec_x, prec_a, prec_aa, prec_ab, prec_a_iso)

    return (F, Z, A, K, prec_a)
