###############################################################################
# Data section
###############################################################################
data = "X.txt"  # name of data matrix X
design = None  # design matrix for observed covariates

initZA=None  # name of factor loading matrix to be initialized, default: None
initSB=None  # name of regression coefficient matrix to be initialized, default: None
initK=7  # initial number of factors if initZA is None
initF=None  # name of score matrix to be initialized. default: None

stdData=False  # standardize data if needed

###############################################################################
# Sampling control parameters
###############################################################################
iteration=1000 # number of MCMC chains
burnin=500 # number of burnin
proposeK=False  # nonparametric IBP if True
updateZA=True  # update ZA matrix if True
updateSB=True # update SB matrix if True
updateF=True  # update F matrix if True
nonGaussianF=False  # sample F using Dirichlet process if True
updateIBPa_fm=True  # update IBP parameter alpha for factor model
updateIBPa_reg=True  # update IBP parameter alpha for regression part
updatePrec_x=True  # update residual precision matrix if True
updatePrec_a=True  # update factor score precision matrix if True
updatePrec_b=True  # update regression coefficient precision matrix if True
prec_x_iso=False  # use isotropic error if True else idiosyncratic error
learn_scale_xb=True  # share power to learn scale when prec_x_iso is False
prec_a_iso=False  # use isotropic error if True else idiosyncratic error
learn_scale_ab=True  # share power to learn scale when prec_a_iso is False
prec_b_iso=False  # use isotropic error if True else idiosyncratic error
learn_scale_bb=True  # share power to learn scale when prec_b_iso is False

###############################################################################
# Model parameters
###############################################################################
prec_x=1  # initial state of precision X (scalar input)
prec_xa=1  # Gamma hyperprior shape parameter for prec_x
prec_xb=1  # Gamma hyperprior rate parameter for prec_x
prec_xb_a=1  # Gamma shape parameter for prec_xb
prec_xb_b=1  # Gamma rate parameter for prec_xb
prec_a=None  # initial state of precision A (scalar input)
prec_aa=1  # Gamma hyperprior shape parameter for prec_a
prec_ab=1  # Gamma hyperprior rate parameter for prec_a
prec_ab_a=1  # Gamma shape parameter for prec_ab
prec_ab_b=1  # Gamma rate parameter for prec_ab
prec_b=None  # initial state of precision A (scalar input)
prec_ba=1  # Gamma hyperprior shape parameter for prec_a
prec_bb=1  # Gamma hyperprior rate parameter for prec_a
prec_bb_a=1  # Gamma shape parameter for prec_ab
prec_bb_b=1  # Gamma rate parameter for prec_ab
fmIBPa=None  # IBP parameter
fmIBPa_a=1  # Gamma hyperprior shape parameter for IBP alpha
fmIBPa_b=1 # Gamma hyperprior rate parameter for IBP alpha
regIBPa=None  # IBP parameter
regIBPa_a=1  # Gamma hyperprior shape parameter for IBP alpha
regIBPa_b=1 # Gamma hyperprior rate parameter for IBP alpha
DPa=1  # Dirichlet process concentration parameter for non-Gaussian F score
DPa_a=1  # Gamma hyperprior shape parameter for DP concentration parameter
DPa_b=1 # Gamma hyperprior rate parameter for DP concentration parameter

###############################################################################
# Optional Arguments
###############################################################################
saveIteration=False  # save simulation output for each itration
printIteration=20  # how often a MCMC iteration is printed
