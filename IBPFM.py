"""Implement sparse factor analysis using IBP prior."""
import numpy as np
import numpy.random as nr
from utils.simulateIBP import simulateIBP
from utils.sampleZA import sampleZA
from utils.sampleSB import sampleSB
from utils.samplePrec_x import samplePrec_x
from utils.samplePrec_a import samplePrec_a
from utils.samplePrec_b import samplePrec_b
from utils.sampleIBPa import sampleIBPa
from utils.logLik import logLik
from utils.saveIter import saveIter

import datetime


def IBPFM(X, iteration, burnin=0, design=None, stdData=False,
          initZA=None, initSB=None, initK=None, initF=None,
          proposeK=True, updateZA=True, updateSB=True,
          updateF=True, nonGaussianF=False,
          updateIBPa_fm=True, updateIBPa_reg=True,
          updatePrec_x=True, updatePrec_a=True, updatePrec_b=True,
          prec_x_iso=False, learn_scale_xb=True,
          prec_a_iso=False, learn_scale_ab=True,
          prec_b_iso=False, learn_scale_bb=True,
          prec_x=None, prec_xa=1, prec_xb=1, prec_xb_a=1, prec_xb_b=1,
          prec_a=None, prec_aa=1, prec_ab=1, prec_ab_a=1, prec_ab_b=1,
          prec_b=None, prec_ba=1, prec_bb=1, prec_bb_a=1, prec_bb_b=1,
          fmIBPa=None, fmIBPa_a=1, fmIBPa_b=1,
          regIBPa=None, regIBPa_a=1, regIBPa_b=1,
          DPa=None, DPa_a=1, DPa_b=1, saveIteration=False,
          printIteration=100):
    """Factor model with IBP prior using Gibbs sampler."""
    # Model: X = BH + (Z o A)F + noise
    # @X: (D x N) data matrix
    # @B: (D x P) regression coefficient for observed covariates
    # @H: (P x N) design matrix for observed covariates
    # @Z: (D x K) binary factor assignment matrix
    # @A: (D x K) factor loading matrix
    # @noise: (D x N) residuals

    # @iteration: # of simulation
    # @data: (D x N) data matrix
    # @design: (D x P) design matrix for covariates

    # OPTIONAL ARGUMENTS
    # @stdData: standardize data if necessary
    # @initZA: initial state of (Z o A) matrix. override initK if it's not None
    # @initK: initial number of features
    # @initF: initial state of F matrix
    # @proposeK: enable non-parametric approach for feature count K
    # @updateZA: update matrix Z and A
    # @updateF: update matrix F
    # @nonGaussianF: use Dirichlet process for F
    # @updatePrec_x: update inverse of residual variance
    # @updatePrec_a: update inverse of factor loading variance
    # #prec_x_iso: use isotropic residual variance?
    # @prec_a_iso: use isotropic factor loading variance?
    # @updateIBPa: update IBP parameter
    # @prec_x: initial state of prec_x (scalar)
    # @prec_xa: Gamma shape parameter for P(prec_x)
    # @prec_xb: Gamma rate parameter for P(prec_x)
    # @prec_xb_a: Gamma shape parameter for P(prec_xb)
    # @prec_xb_b: Gamma rate parameter for P(prec_xb)
    # @prec_a: initial state of prec_a (scalar)
    # @prec_aa: Gamma shape parameter for P(prec_a)
    # @prec_ab: Gamma rate parameter for P(prec_a)
    # @prec_ab_a: Gamma shape parameter for P(prec_ab)
    # @prec_ab_b: Gamma rate parameter for P(prec_ab)
    # @fmIBPa: IBP alpha Parameter for factor model part
    # @fmIBPa_a: Gamma shape parameter for P(fmIBPa)
    # @fmIBPa_b: Gamma rate parameter for P(fmIBPa)
    # @regIBPa: IBP alpha Parameter for regression part
    # @regIBPa_a: Gamma shape parameter for P(regIBPa)
    # @regIBPa_b: Gamma rate parameter for P(regIBPa)
    # @DPa: concentration parameter for Dirichlet process
    # @DPa_b: Gamma shape parameter for P(DPa)
    # @DPa_b: Gamma rate parameter for P(DPa)
    # @saveIteration: save output for each iteration as file

    D, N = X.shape

    # Create a matrix with missing indicators
    Xmask = np.isnan(X).astype(np.int)

    if stdData:
        X = (X - np.nanmean(X, axis=1, keepdims=True)) /\
            np.nanstd(X, axis=1, keepdims=True)

    # Initialize noise variance from Gamma prior
    if prec_x is None:
        prec_x = np.ones(D) * nr.gamma(prec_xa, 1./prec_xb)
    else:
        prec_x = np.ones(D) * prec_x

    # Initialize IBP parameter alpha from Gamma prior
    if fmIBPa is None:
        fmIBPa = nr.gamma(fmIBPa_a, fmIBPa_b)
    if design is not None and regIBPa is None:
        regIBPa = nr.gamma(regIBPa_a, regIBPa_b)

    if initZA is None:
        # Generate binary feature assignment matrix Z
        if initK is not None:
            K = initK
            Z = nr.binomial(1, 0.5, (D, K))
        else:
            Z = simulateIBP(regIBPa, D)
            Z = Z.astype(np.int)
            K = Z.shape[1]

        # Initialize feature loading variance from Gamma prior
        if prec_a is None:
            prec_a = np.ones(K) * nr.gamma(prec_aa, 1./prec_ab)
        else:
            prec_a = np.ones(K) * prec_a

        # Simulate feature loading matrix A based on N(A_dk | 0, sigma_a)
        A = np.copy(Z)
        A = A.astype(np.float)
        for (d, k) in zip(*A.nonzero()):
            A[d, k] = nr.normal(0, np.sqrt(1/prec_a[k]))

    else:
        A = initZA.astype(np.float)
        Z = (A != 0).astype(np.int)

        K = Z.shape[1]

        # Initialize feature loading variance from Gamma prior
        if prec_a is None:
            prec_a = np.ones(K) * nr.gamma(prec_aa, 1./prec_ab)
        else:
            prec_a = np.ones(K) * prec_a

    # Check A is float type and Z is integer type
    assert(np.issubsctype(A, np.float) and np.issubsctype(Z, np.int))

    # Initialize feature score matrix F
    if initF is None:
        if nonGaussianF:
            from utils.sampleF_DP import sampleF
            if DPa is None:
                # Draw DP concentration parameter
                DPa = nr.gamma(DPa_a, 1./DPa_b)
            while True:
                # Redraw DP concentration parameter until E(# of cluster) > 0
                J = int(DPa*np.log(N))
                if (J > 0):
                    break
                else:
                    DPa = nr.gamma(DPa_a, 1./DPa_b)
            # Initialize cluster indicators for samples
            clus_ind = nr.choice(range(J), size=N)
            # Initialize unique factor vectors
            clus_theta = nr.normal(0, 1, (K, J))
            F = np.empty((K, N)) * np.nan
            for i in range(N):
                F[:, i] = clus_theta[:, clus_ind[i]]
        else:
            from utils.sampleF import sampleF
            F = nr.normal(0, 1, size=(K, N))
    else:
        F = initF.astype(np.float)
        assert(initF.shape == (K, N))
        if nonGaussianF:
            from utils.sampleF_DP import sampleF
            if (DPa is None):
                # Draw DP concentration parameter
                DPa = nr.gamma(DPa_a, 1./DPa_b)

            # Extract unique score vectors and class indicators
            clus_theta = np.vstack({tuple(row) for row in F.T}).T
            assert(clus_theta.shape[0] == K)
            J = clus_theta.shape[1]
            clus_ind = np.array([], dtype=np.int)
            for i in range(N):
                label = np.where((F[:, [i]] == clus_theta).all(axis=0))
                clus_ind = np.append(clus_ind, label)
            assert(len(clus_ind) == N)
        else:
            from utils.sampleF import sampleF

    # Impute missing values if they exist
    if np.sum(Xmask) > 0:
        Xtemp = np.dot(A, F)
        X[Xmask == 1] = Xtemp[Xmask == 1]
        nonMissing = X[Xmask == 0]

    # Regression components for design and control variables
    if design is not None:
        H = design
        P = H.shape[0]
        assert(H.shape[1] == N)

        if initSB is None:
            # Generate binary feature assignment matrix Z
            intercept = np.ones(D).reshape(D, 1)
            S1 = nr.binomial(1, 0.5, (D, P-1))
            S = np.hstack((intercept, S1)).astype(np.int)

            # Initialize feature loading variance from Gamma prior
            if prec_b is None:
                prec_b = np.ones(P) * nr.gamma(prec_ba, 1./prec_bb)
            else:
                prec_b = np.ones(P) * prec_b

            # Simulate coefficient matrix B based on N(B_dk | 0, sigma_b)
            B = np.copy(S)
            B = S.astype(np.float)
            mean = np.nanmean(X, axis=1)
            for (d, p) in zip(*B.nonzero()):
                if p == 0:
                    # Start intercept with variable mean
                    B[d, p] = mean[d]
                else:
                    B[d, p] = nr.normal(0, np.sqrt(1/prec_b[p]))
        else:
            B = initZA.astype(np.float)
            S = (B != 0).astype(np.int)
            assert(B.shape == (D, P))

            # Initialize feature loading variance from Gamma prior
            if prec_b is None:
                prec_b = np.ones(P) * nr.gamma(prec_ba, 1./prec_bb)
            else:
                prec_b = np.ones(P) * prec_b

        # Check B is float type and S is integer type
        assert(np.issubsctype(B, np.float) and np.issubsctype(S, np.int))

    for s in range(iteration):
        # Save initial parameters
        if (s == 0):
            K_save = K
            fmIBPa_save = fmIBPa
            if design is not None:
                regIBPa_save = regIBPa
            psi_x_save = 1./prec_x
            if nonGaussianF:
                DPa_save = DPa

            if design is not None:
                loglikelihood = logLik(X, F, A, prec_x, N, D, prec_x_iso, H, B)
            else:
                loglikelihood = logLik(X, F, A, prec_x, N, D, prec_x_iso,
                                       H=None, B=None)

            logLik_save = loglikelihood

            if proposeK is False:
                tau_a_save = 1./prec_a
                Z_sum = np.zeros((D, K))
                A_sum = np.zeros((D, K))
                F_sum = np.zeros((K, N))
                if design is not None:
                    S_sum = np.zeros((D, P))
                    B_sum = np.zeros((D, P))

            time = datetime.datetime.now()
            print ("=========================================================")
            print ("Started at " + str(time))
            print ("Data shape: observations = %d\t variables = %d" % (N, D))
            print ("K = %d\tIBP_alpha = %.3f" % (K, fmIBPa))
            print ("=========================================================")

        # Update coefficient matrix
        if design is not None:
            X_reg = X - np.dot(A, F)
            if updateSB:
                (S, B) = sampleSB(X_reg, H, S, B, prec_x, prec_b, regIBPa, P,
                                  N, D, prec_ba, prec_bb, prec_b_iso)
            X_fm = X - np.dot(B, H)

            # Update IBP parameter for regression part
            if updateIBPa_reg:
                regIBPa = sampleIBPa(regIBPa_a, regIBPa_b, P, D)

            # Update coefficient variance
            if updatePrec_b:
                if (learn_scale_bb and not prec_b_iso):
                    from utils.samplePrec_b import samplePrec_bb
                    prec_bb = samplePrec_bb(P, prec_b, prec_ba, prec_bb,
                                            prec_bb_a, prec_bb_b)

                prec_b = samplePrec_b(X_reg, S, B, P, prec_ba,
                                      prec_bb, prec_b_iso)
        else:
            X_fm = X

        # Update factor assignment matrix Z and factor loading matrix A
        if updateZA:
            (F, Z, A, K, prec_a) = sampleZA(X_fm, F, Z, A, prec_x, prec_a,
                                            fmIBPa, K, N, D, proposeK,
                                            prec_aa, prec_ab, prec_a_iso)

        # Update factor score matrix
        if updateF and nonGaussianF:
            (F, clus_ind, clus_theta, J, DPa) = sampleF(X_fm, F, A, prec_x,
                                                        clus_ind, clus_theta,
                                                        J, N, D, K,
                                                        DPa, DPa_a, DPa_b)

        if updateF and not nonGaussianF:
            F = sampleF(X_fm, A, prec_x, N, D, K)

        # Update factor loading variance
        if updatePrec_a:
            if (learn_scale_ab and not prec_a_iso):
                from utils.samplePrec_a import samplePrec_ab
                prec_ab = samplePrec_ab(K, prec_a, prec_aa, prec_ab,
                                        prec_ab_a, prec_ab_b)

            prec_a = samplePrec_a(X_fm, Z, A, K,
                                  prec_aa, prec_ab, prec_a_iso)

        # Update IBP parameter for factor model part
        if updateIBPa_fm:
            fmIBPa = sampleIBPa(fmIBPa_a, fmIBPa_b, K, D)

        # Update residual variance
        if updatePrec_x:
            if (learn_scale_xb and not prec_x_iso):
                from utils.samplePrec_x import samplePrec_xb
                prec_xb = samplePrec_xb(D, prec_x, prec_xa, prec_xb,
                                        prec_xb_a, prec_xb_b)

            if design is not None:
                residue = X - np.dot(B, H) - np.dot(A, F)
            else:
                residue = X - np.dot(A, F)
            prec_x = samplePrec_x(residue, N, D, prec_xa, prec_xb, prec_x_iso)

        # Update missing values based on posterior distribution
        if np.sum(Xmask > 0):
            # Predictive mean
            if design is not None:
                Xpred = np.dot(B, H) + np.dot(A, F)
            else:
                Xpred = np.dot(A, F)

            # Add noise
            covNoise = np.diag(1./prec_x)
            noise = nr.multivariate_normal(np.zeros(D), covNoise, N).T
            Xpred += noise

            # Update missing values
            X[Xmask == 1] = Xpred[Xmask == 1]
            assert(all(nonMissing == X[Xmask == 0]))

        if design is not None:
            loglikelihood = logLik(X, F, A, prec_x, N, D, prec_x_iso, H, B)
        else:
            loglikelihood = logLik(X, F, A, prec_x, N, D, prec_x_iso,
                                   H=None, B=None)

        if (s+1) % printIteration == 0:
            print ("Iteration %d: K = %d\tIBP_alpha = %.3f\tlogLik= %.3f" %
                   ((s+1), K, fmIBPa, loglikelihood))

        # Save parameters for each iteration
        K_save = np.append(K_save, K)
        fmIBPa_save = np.append(fmIBPa_save, fmIBPa)
        if design is not None:
            regIBPa_save = np.append(regIBPa_save, regIBPa)
        psi_x_save = np.vstack((psi_x_save, 1./prec_x))
        if nonGaussianF:
            DPa_save = np.append(DPa_save, DPa)
        logLik_save = np.append(logLik_save, loglikelihood)

        if proposeK is False:
            tau_a_save = np.vstack((tau_a_save, 1./prec_a))
            # Accumulate Z, A, F to calculate posterior mean
            if (s >= burnin):
                Z_sum = Z_sum + Z
                A_sum = A_sum + A
                F_sum = F_sum + F
                if design is not None:
                    S_sum = S_sum + S
                    B_sum = B_sum + B

        if saveIteration and s >= burnin:
            saveIter(s, Z, A, F, prec_x, prec_a)

    fmIBPa_mean = np.mean(fmIBPa_save[(burnin+1):])
    psi_mean = psi_x_save[(burnin+1):, :].mean(axis=0)
    np.savetxt("mIBPalpha_Fm.txt", np.array([fmIBPa_mean]), delimiter="\t")
    np.savetxt("mPsi.txt", psi_mean.reshape(1, psi_mean.shape[0]),
               delimiter="\t")

    np.savetxt("logLik.txt", logLik_save, delimiter="\t")

    if proposeK is False:
        NMCsample = iteration - burnin
        Z_mean = Z_sum.astype(np.float) / NMCsample
        A_mean = A_sum / NMCsample
        F_mean = F_sum / NMCsample
        tau_mean = tau_a_save[(burnin+1):, :].mean(axis=0)
        np.savetxt("mZ.txt", Z_mean, delimiter="\t")
        np.savetxt("mA.txt", A_mean, delimiter="\t")
        np.savetxt("mF.txt", F_mean, delimiter="\t")
        np.savetxt("mTau.txt", tau_mean.reshape(1, tau_mean.shape[0]),
                   delimiter="\t")
        if nonGaussianF:
            DPa_mean = np.mean(DPa_save[(burnin+1):])
            np.savetxt("mDPalpha.txt", np.array([DPa_mean]), delimiter="\t")

    else:
        np.savetxt("K.txt", K_save, delimiter="\t")

    if design is not None:
        regIBPa_mean = np.mean(regIBPa_save[(burnin+1):])
        np.savetxt("mRegIBPalpha_Reg.txt", np.array([regIBPa_mean]),
                   delimiter="\t")

        NMCsample = iteration - burnin
        S_mean = S_sum / NMCsample
        B_mean = B_sum / NMCsample
        np.savetxt("mS.txt", S_mean, delimiter="\t")
        np.savetxt("mB.txt", B_mean, delimiter="\t")

    return


if __name__ == "__main__":
    # print "Test run success"
    import MCMCsetting as mc
    import cProfile
    X = np.genfromtxt(fname=mc.data, dtype=None,
                      delimiter="\t", missing_values="")

    if mc.design is not None and mc.initZA is not None:
        ZA = np.genfromtxt(fname=mc.initZA, dtype=None, delimiter="\t")
    else:
        ZA = None

    if mc.initSB is not None:
        SB = np.genfromtxt(fname=mc.initSB, dtype=None, delimiter="\t")
    else:
        SB = None

    if mc.initF is not None:
        F = np.genfromtxt(fname=mc.initF, dtype=None, delimiter="\t")
    else:
        F = None

    pr = cProfile.Profile()  # profiling
    pr.enable()
    IBPFM(X=X, iteration=mc.iteration, burnin=mc.burnin,
          design=mc.design, stdData=mc.stdData,
          initZA=ZA, initSB=SB, initK=mc.initK, initF=F,
          proposeK=mc.proposeK, updateZA=mc.updateZA, updateSB=mc.updateSB,
          updateF=mc.updateF, nonGaussianF=mc.nonGaussianF,
          updateIBPa_fm=mc.updateIBPa_fm, updateIBPa_reg=mc.updateIBPa_reg,
          updatePrec_x=mc.updatePrec_x, updatePrec_a=mc.updatePrec_a,
          updatePrec_b=mc.updatePrec_b,
          prec_x_iso=mc.prec_x_iso, learn_scale_xb=mc.learn_scale_xb,
          prec_a_iso=mc.prec_a_iso, learn_scale_ab=mc.learn_scale_ab,
          prec_b_iso=mc.prec_b_iso, learn_scale_bb=mc.learn_scale_bb,
          prec_x=mc.prec_x, prec_xa=mc.prec_xa, prec_xb=mc.prec_xb,
          prec_xb_a=mc.prec_xb_a, prec_xb_b=mc.prec_xb_b,
          prec_a=mc.prec_a, prec_aa=mc.prec_aa, prec_ab=mc.prec_ab,
          prec_ab_a=mc.prec_ab_a, prec_ab_b=mc.prec_ab_b,
          prec_b=mc.prec_b, prec_ba=mc.prec_ba, prec_bb=mc.prec_bb,
          prec_bb_a=mc.prec_bb_a, prec_bb_b=mc.prec_bb_b,
          fmIBPa=mc.fmIBPa, fmIBPa_a=mc.fmIBPa_a, fmIBPa_b=mc.fmIBPa_b,
          regIBPa=mc.regIBPa, regIBPa_a=mc.regIBPa_a, regIBPa_b=mc.regIBPa_b,
          DPa=mc.DPa, DPa_a=mc.DPa_a, DPa_b=mc.DPa_b,
          saveIteration=mc.saveIteration,
          printIteration=mc.printIteration)
    pr.disable()
    print("")
    print("MCMC Profiling Output: ")
    print("")
    pr.print_stats(sort='time')
