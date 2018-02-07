"""Save MCMC output for each iteration."""
import numpy as np


def saveIter(s, Z, A, F, prec_x, prec_a):
    """Define function to save MCMC when required."""
    Z_name = "Z" + str(s+1) + ".txt"
    A_name = "A" + str(s+1) + ".txt"
    F_name = "F" + str(s+1) + ".txt"
    psi_name = "Psi" + str(s+1) + ".txt"
    tau_name = "Tau" + str(s+1) + ".txt"
    np.savetxt(Z_name, Z, delimiter="\t")
    np.savetxt(A_name, A, delimiter="\t")
    np.savetxt(F_name, F, delimiter="\t")
    np.savetxt(psi_name, 1./prec_x, delimiter="\t")
    np.savetxt(tau_name, 1./prec_a, delimiter="\t")

    return
