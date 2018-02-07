"""Update coefficients for observed covariates and design factors."""
import numpy as np
import numpy.random as nr


def sampleSB(X, H, S, B, prec_x, prec_b, IBPa_reg, P, N, D,
             prec_ba, prec_bb, prec_b_iso):
    """Update B_dp using finite IBP prior."""
    # Check matrix type: S - int, B - float
    assert(np.issubsctype(S, np.int) and np.issubsctype(B, np.float))

    # Calculate vector H_p * H_p.T
    HH = np.dot(H, H.T).diagonal()
    E = X - np.dot(B, H)

    for d in range(D):
        for p in range(P):
            m_p = np.sum(S[:, p]) - S[d, p]

            # Evaluating error assuming B_dp = 0
            # Since E mat already subtracted an error attributed to B_dp,
            # we add it back here
            if S[d, p]:
                ep = E[[d], :] + B[d, p]*H[[p], :]
            else:
                ep = E[[d], :]

            # Calculate Lambda = prec_x[d]*H_p*H_p.T + prec_b[p]
            Lambda = prec_x[d] * HH[p] + prec_b[p]
            Mu = (prec_x[d] * np.dot(H[[p], :], ep.T)) / Lambda

            log_prior_ratio = np.log(m_p+IBPa_reg/P) - np.log(D-m_p)

            # Calculate log likelihood ratio P(X|S_dp=1,-) / P(X|S_dp=0,-)
            log_ll_ratio = 0.5*(np.log(prec_b[p])-np.log(Lambda))
            log_ll_ratio += 0.5*(Lambda*(Mu**2))

            # Calculate probability ratio
            log_prob_ratio = log_prior_ratio + log_ll_ratio
            prob_ratio = np.exp(-log_prob_ratio)

            # Calculate probability
            p1 = 1./(1 + prob_ratio)
            assert(not np.isnan(p1).any())

            # Update S_dp and B_dp
            S[d, p] = 1 if nr.uniform() < p1 else 0
            if S[d, p]:
                sd = np.sqrt(1./Lambda)
                B[d, p] = nr.normal(Mu, sd)
            else:
                B[d, p] = 0

            E[d, :] = ep - B[d, p] * H[p, :]

    return (S, B)
