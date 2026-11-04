# Exercise exr-ren-kk-unify: the cutoff at which alpha_1 = alpha_2 with Kaluza-Klein towers (Dienes-Dudas-Gherghetta eq. 3.13).
from scipy.optimize import brentq
import numpy as np
MZ = 91.17; ainv = [98.29 * 3 / 5, 29.61, 8.5]; b = [33 / 5, 1, -3]; bt = [3 / 5, -3, -6]
def a(i, L, mu0, d):
    X = float(pi^(d / 2) / gamma(1 + d / 2)); r = ainv[i] - b[i] / (2 * np.pi) * np.log(L / MZ)
    if L > mu0: r += bt[i] / (2 * np.pi) * np.log(L / mu0) - bt[i] * X / (2 * np.pi * d) * ((L / mu0)**d - 1)
    return r
for mu0 in [1e5, 1e8, 1e11]:
    for d in [1, 2]:
        L = brentq(lambda x: a(0, x, mu0, d) - a(1, x, mu0, d), mu0 * 1.0001, mu0 * 1e4)
        print("mu0 = %.0e, delta = %d: Lambda/mu0 = %5.2f, alpha_1^-1 = alpha_2^-1 = %.2f, alpha_3^-1 = %.2f" % (mu0, d, L / mu0, a(0, L, mu0, d), a(2, L, mu0, d)))
