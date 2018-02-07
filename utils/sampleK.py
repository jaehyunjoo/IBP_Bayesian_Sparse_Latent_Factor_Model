"""Update the number of features K."""
import numpy as np
import numpy.random as nr


def sampleK(d, poiMu, Knew,
            N, D, K, X, F, Z, A, E, FF,
            prec_x, prec_a, prec_aa, prec_ab, prec_a_iso):
    """Propose new feature count K using M-H step."""
    if prec_a_iso and Knew > 0:
        prec_a_new = np.ones(Knew)*prec_a[1]

    if not prec_a_iso and Knew > 0:
        # sample from Gamma prior
        prec_a_new = nr.gamma(prec_aa, 1./prec_ab, Knew)

    # Find singleton clusters
    m_neg_d = np.sum(Z[d, :], axis=0) - Z[d, :]
    singletons = [kd for kd in range(K) if m_neg_d[kd] == 0]
    current_kappa = len(singletons)  # current number of singletons
    assert(isinstance(current_kappa, int))

    # Calculate error E_dn assuming all singletons for row d switched off
    Ad = np.copy(A[d, :])
    Ad[singletons] = 0
    Ed = X[[d], :] - np.dot(Ad, F)

    current_a = A[d, singletons].reshape(current_kappa, 1)

    if current_kappa > 0:
        M = prec_x[d] * np.dot(current_a, current_a.T) + np.eye(current_kappa)
        M_inv = np.linalg.inv(M)
        m_n = prec_x[d] * np.dot(np.dot(M_inv, current_a), Ed)

        log_prop_de = -0.5 * N * np.log(np.linalg.det(M_inv))
        log_prop_de += 0.5 * np.trace(np.dot(m_n.T, np.dot(M, m_n)))
        log_prop_de += (Knew)*np.log(poiMu)

    if Knew > 0:
        a = (nr.standard_normal(Knew)*np.sqrt(1./prec_a_new)).reshape(Knew, 1)
        M_star = prec_x[d]*np.dot(a, a.T) + np.eye(Knew)
        M_star_inv = np.linalg.inv(M_star)
        m_star_n = prec_x[d]*np.dot(np.dot(M_star_inv, a), Ed)

        log_prop_nu = -0.5*N*np.log(np.linalg.det(M_star_inv))
        log_prop_nu += 0.5*np.trace(np.dot(m_star_n.T, np.dot(M_star, m_star_n)))
        log_prop_nu += (current_kappa)*np.log(poiMu)

    if current_kappa > 0 and Knew > 0.5:
        log_prop = log_prop_nu - log_prop_de
        prop = np.exp(log_prop)

    if current_kappa > 0 and Knew == 0:
        prop = 0

    if current_kappa == 0 and Knew >= 0:
        prop = 1

    if nr.uniform() < prop:
        if (current_kappa > 0):
            # Update error term
            # We are deleting singleton features, so add beck regarding these.
            E[d, :] = E[d, :] + np.dot(A[d, singletons], F[singletons, :])

        # Delete redundant features
        Z[d, singletons] = 0

        m = np.sum(Z, axis=0)
        idx = [kd for kd in range(K) if m[kd] == 0]  # just refresh
        Z = np.delete(Z, idx, axis=1)
        A = np.delete(A, idx, axis=1)
        F = np.delete(F, idx, axis=0)
        FF = np.delete(FF, idx)
        prec_a = np.delete(prec_a, idx)
        K = len(prec_a)
        assert(Z.shape[1] == A.shape[1] == F.shape[0] == K)

        if Knew > 0:
            Z = np.hstack((Z, np.zeros((D, Knew)))).astype(np.int)
            Z[d, range(-Knew, 0)] = 1
            M_star_inv_L = np.linalg.cholesky(M_star_inv)
            normal_sample = nr.standard_normal((Knew, N))
            Fnew = m_star_n + np.dot(M_star_inv_L, normal_sample)
            F = np.vstack((F, Fnew))
            FFnew = np.dot(Fnew, Fnew.T).diagonal()
            FF = np.append(FF, FFnew)

            A = np.hstack((A, np.zeros((D, Knew))))
            A[d, range(-Knew, 0)] = a.T
            prec_a = np.append(prec_a, prec_a_new)

            Kold = K
            K += Knew
            assert(Z.shape[1] == A.shape[1] == F.shape[0] == len(prec_a) == K)

            # Update error term based on new values
            # We are adding new features, so subtract regarding these.
            E[d, :] = E[d, :] - np.dot(a.T, Fnew)

            for k in range(Kold, K):
                ek = E[d, :] + A[d, k] * F[k, :]  # assume A_dk = 0
                precision = prec_x[d] * FF[k] + prec_a[k]
                mu = (prec_x[d] * np.dot(F[k, :], ek.T)) / precision
                sd = np.sqrt(1./precision)
                A[d, k] = nr.normal(mu, sd)

    E = X - np.dot(A, F)
    assert(np.issubsctype(Z, np.int))

    return (Z, A, F, FF, prec_a, E, K)
