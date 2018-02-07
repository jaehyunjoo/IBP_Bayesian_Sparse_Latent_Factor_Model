"""Update factor score matrix F by using Dirichlet process."""
import numpy as np
import numpy.random as nr


def sampleF(X, F, A, prec_x, clus_ind, clus_theta, J, N, D, K, DPa,
            DPa_a, DPa_b):
    """Relaxation of Gaussian assumption using Dirichlet process."""
    # @clus_ind 1xN vector that represents membership 0:s for each obs
    # @clus_theta: KxJ (1:J) distinct factor vector theta

    # Calculate Psi
    psi = np.diag(1./prec_x)
    psi_inv = np.diag(prec_x)

    # Calculate membership count
    n_j = np.bincount(clus_ind)  # much faster than np.unique

    for i in range(N):
        # Calculate n_j: how many times we've seen theta_j, excluding i-th obs
        oldC = clus_ind[i]
        n_j[oldC] -= 1

        # Remove empty clusters
        if n_j[oldC] == 0:
            J -= 1
            n_j = np.delete(n_j, oldC)
            clus_ind[clus_ind > oldC] -= 1
            clus_theta = np.delete(clus_theta, oldC, axis=1)

        # Conditional probability of assigning i-th obs to existing clusters
        E_i = X[:, [i]]  # X is actually (X - BH) in case covariates exist
        ll_ex_prob = np.zeros(J)
        for j in range(J):
            # P(assigning j group) is proportional to n_j*N(E_i|A*theta_j, Psi)
            ex_mean = np.dot(A, clus_theta[:, [j]])
            mat = E_i - ex_mean
            ll_ex_prob[j] = np.log(n_j[j])
            ll_ex_prob[j] -= 0.5 * D * np.log(2*np.pi)
            ll_ex_prob[j] -= 0.5 * np.log(np.prod(1./prec_x))
            ll_ex_prob[j] -= 0.5 * np.dot(mat.T, np.dot(psi_inv, mat))

        # Conditional probability of drawing new cluster
        # P(assigning new cluster) is proportional to DPa*N(E_i|0, A'A+Psi)
        newC_var = np.dot(A, A.T) + psi
        ll_new_prob = np.log(DPa)
        ll_new_prob -= 0.5 * D * np.log(2*np.pi)
        ll_new_prob -= 0.5 * np.log(np.linalg.det(newC_var))
        ll_new_prob -= 0.5 * np.dot(E_i.T,
                                    np.dot(np.linalg.inv(newC_var), E_i))

        # Full set of conditional probabilities
        ll_cond_prob = np.append(ll_ex_prob, ll_new_prob)
        assert(all(np.isfinite(ll_cond_prob)))

        cond_prob = np.exp(ll_cond_prob - max(ll_cond_prob))  # rescale
        cond_prob = cond_prob / np.sum(cond_prob)

        # Sample new membership for i-th obs
        assert(J+1 == len(cond_prob))
        newC = nr.choice(range(J+1), p=cond_prob)  # draw sample from 0 to J
        clus_ind[i] = newC
        assert(0 <= newC <= J)

        # If newC > J, sample new factor vector theta_(j+1)
        if newC == J:
            # Draw new unique factor score vector
            # it sampled from N(m_i, M) where
            # M^-1 = I + A'(Psi^-1)A, m_i = MA'(Psi^-1)E_i
            M_inv = np.eye(K) + np.dot(A.T, np.dot(psi_inv, A))
            M = np.linalg.inv(M_inv)
            m_i = np.dot(M, np.dot(A.T, np.dot(psi_inv, E_i)))
            L = np.linalg.cholesky(M)
            normal_sample = nr.standard_normal((K, 1))
            new_vector = m_i + np.dot(L, normal_sample)
            clus_theta = np.hstack((clus_theta, new_vector))
            J += 1
            n_j = np.append(n_j, 0)

        # Update membership count
        n_j[newC] += 1

        # Update factor score vector
        F[:, i] = np.copy(clus_theta[:, newC])

    # Calculate n_j after updating full set of configuration indicators
    # Not strictly necessary
    n_j_test = np.bincount(clus_ind)
    assert(all(n_j == n_j_test))

    # Update unique factor vectors (clus_theta)
    for j in range(J):
        # Implied conditional posterior P(theta_j | -)
        # ~ N(theta_j | t_j, T_j) where
        # T_j^-1 = I + n_j*A'(Psi^-1)A,
        # t_j = T_jA'(Psi^-1)(np.sum(X[, C==j], axis=1))
        theta_var_inv = np.eye(K) + n_j[j]*np.dot(A.T, np.dot(psi_inv, A))
        theta_var = np.linalg.inv(theta_var_inv)

        sum_E_i = np.sum(X[:, clus_ind == j], axis=1, keepdims=True)
        theta_mean = np.dot(theta_var, np.dot(A.T, np.dot(psi_inv, sum_E_i)))
        L_chol = np.linalg.cholesky(theta_var)
        z_sample = nr.standard_normal(size=(K, 1))
        new_theta_j = theta_mean + np.dot(L_chol, z_sample)

        clus_theta[:, [j]] = new_theta_j

    # Update Dirichlet process parameter DPa
    # from Escobar and West, 1995
    # eta = nr.beta(DPa + 1, N)
    eta = nr.beta(DPa+1, N)
    ratio = (DPa_a+J-1) / (N*(DPa_b-np.log(eta)))
    Pi = ratio / (1+ratio)
    if nr.uniform() < Pi:
        DPa = nr.gamma(DPa_a+J, 1./(DPa_b-np.log(eta)))
    else:
        DPa = nr.gamma(DPa_a+J-1, 1./(DPa_b-np.log(eta)))

    return (F, clus_ind, clus_theta, J, DPa)
