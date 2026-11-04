# Exercise exr-sr-exp-closed-form: e^X = cosh(lambda) 1 + sinh(lambda)/lambda X for traceless 2x2 X with lambda^2 = -det X.
set_random_seed(5)
X = matrix(CDF, 2, 2, [CDF.random_element(), CDF.random_element(), CDF.random_element(), 0]); X[1, 1] = -X[0, 0]
lam = sqrt(-X.det())
F = cosh(lam)*identity_matrix(CDF, 2) + sinh(lam)/lam*X
print("max |exp(X) - closed form| =", max(abs(a - b) for a, b in zip(X.exp().list(), F.list())))
print("trace of exp(X) = %s,  2 cosh(lambda) = %s" % (X.exp().trace(), 2*cosh(lam)))
import numpy as np
from scipy.linalg import logm
L = logm(-np.array([[1, 1], [0, 1]], dtype=complex))
print("principal logarithm of A0 = -[[1,1],[0,1]] has trace", np.round(np.trace(L), 12), "(not traceless, as the theorem predicts)")
