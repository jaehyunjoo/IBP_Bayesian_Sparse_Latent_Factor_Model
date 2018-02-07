"""Test IBP-FM using 6x6 Image example from Griffiths and Ghahramani 2005."""
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from IBPFM import IBPFM
from utils.scaledimage import scaledimage

# ------------------------------------------------------------------------------
# Generate image data from the known features
# ------------------------------------------------------------------------------
feature1 = np.array([[0, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
feature2 = np.array([[0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
feature3 = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0]])
feature4 = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1, 0]])

D = 36  # number of variables
f1 = feature1.reshape(D, 1)
f2 = feature2.reshape(D, 1)
f3 = feature3.reshape(D, 1)
f4 = feature4.reshape(D, 1)

N = 500  # number of data points
K = 4  # number of latent features
sig_x_true = 0.2  # true residual variance

# Construct feature assignment matrix and mixture weight matrix
A = Z = np.hstack((f1, f2, f3, f4))

# Construct feature score matrix F
FS = nr.standard_normal((K, N))


# Add covariates: intercept + binary factor + continuous var
P = 2
intercept = np.ones(N)
binary = np.ones(N/2)
binary = np.append(binary, np.zeros(N/2))
# cont = nr.normal(0, 0.2, N)
cont = np.ones(N)
H = np.vstack((intercept, binary))  # design matrix
B1 = ((np.arange(D)+1) - np.mean(np.arange(D)+1)) / np.std(np.arange(D)+1)
B2 = np.hstack((np.repeat(-1, 12), np.repeat(0, 12), np.repeat(1, 12)))
# B3 = np.hstack((np.repeat(-1, 18), np.repeat(1, 18)))
# B3 = np.linspace(-.5, .5, num=36)
B = np.vstack((B1, B2)).T

# Generate noisy image
X = np.dot(A, FS)
X += np.dot(B, H)  # skip this step if you don't want a covariate adjustment
X += nr.normal(0, sig_x_true, (D, N))  # add noise
assert(np.linalg.matrix_rank(X) == D)

# ------------------------------------------------------------------------------
# Do test
# ------------------------------------------------------------------------------

IBPFM(X=X, iteration=2000, burnin=1500, design=H, stdData=False,
      initZA=None, initF=None, nonGaussianF=False,
      proposeK=False, prec_x_iso=False, prec_a_iso=False,
      learn_scale_xb=True, learn_scale_ab=True,
      saveIteration=False, DPa=1, initK=4, printIteration=10)

# # Save true latent feature plot
trueWeights = A.T
# (orig, sub) = plt.subplots(1, 4)
# for sa in sub.flatten():
#     sa.set_visible(False)

# orig.suptitle('True Latent Features')

# for (i, true) in enumerate(trueWeights):
#     ax = sub[i]
#     ax.set_visible(True)
#     scaledimage(true.reshape(6, 6), pixwidth=3, ax=ax)

# orig.set_size_inches(13, 3)
# orig.savefig('Demo_True_Latent_Features.png')
# plt.close()

# # Save some of example figures from data X
# examples = X.T[0:4, :]
# (ex, sub) = plt.subplots(1, 4)
# for sa in sub.flatten():
#     sa.set_visible(False)
# ex.suptitle('Some of Images')
# for (i, true) in enumerate(examples):
#     ax = sub[i]
#     ax.set_visible(True)
#     scaledimage(true.reshape(6, 6), pixwidth=3, ax=ax)

# ex.set_size_inches(13, 3)
# ex.savefig('Demo_Image_Examples.png')
# plt.close()

mA = np.genfromtxt(fname="mA.txt", dtype=None, delimiter="\t")
postA = mA.T

postA_row = postA.shape[0]
for i in range(postA_row):
    cur_row = postA[i, :].tolist()
    abs_row = [abs(j) for j in cur_row]
    max_index = abs_row.index(max(abs_row))
    if cur_row[max_index] < 0:
        postA[i, :] = -np.array(cur_row)

K = max(len(trueWeights), len(postA))
(fig, subaxes) = plt.subplots(2, K)
for sa in subaxes.flatten():
    sa.set_visible(False)
fig.suptitle('Ground truth (top) vs learned features (bottom)')
for (idx, trueFactor) in enumerate(trueWeights):
    ax = subaxes[0, idx]
    ax.set_visible(True)
    scaledimage(trueFactor.reshape(6, 6),
                pixwidth=3, ax=ax)
for (idx, learnedFactor) in enumerate(postA):
    ax = subaxes[1, idx]
    scaledimage(learnedFactor.reshape(6, 6),
                pixwidth=3, ax=ax)
    ax.set_visible(True)
fig.savefig("Demo_Learned_Features_from_Images.png")

# Save true regression coefficients plot
trueWeights = B.T
# (orig, sub) = plt.subplots(1, 2)
# for sa in sub.flatten():
#     sa.set_visible(False)

# orig.suptitle('True Regression Coefficients')

# for (i, true) in enumerate(trueWeights):
#     ax = sub[i]
#     ax.set_visible(True)
#     scaledimage(true.reshape(6, 6), pixwidth=3, ax=ax)

# orig.set_size_inches(13, 3)
# orig.savefig('Demo_True_Regression_Coefficients.png')
# plt.close()

mB = np.genfromtxt(fname="mB.txt", dtype=None, delimiter="\t")
postB = mB.T

P = max(len(trueWeights), len(postB))
(fig, subaxes) = plt.subplots(2, P)
for sa in subaxes.flatten():
    sa.set_visible(False)
fig.suptitle('Ground truth (top) vs learned coefficients (bottom)')
for (idx, trueFactor) in enumerate(trueWeights):
    ax = subaxes[0, idx]
    ax.set_visible(True)
    scaledimage(trueFactor.reshape(6, 6),
                pixwidth=3, ax=ax)
for (idx, learnedFactor) in enumerate(postB):
    ax = subaxes[1, idx]
    scaledimage(learnedFactor.reshape(6, 6),
                pixwidth=3, ax=ax)
    ax.set_visible(True)
fig.savefig("Demo_Learned_Regression_Coefficients.png")
