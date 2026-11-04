# Exercise exr-lgt-cl-quartic: 1 - (1/N) Re Tr exp(M) for M in su(N): the quadratic term and the absence of a cubic one.
import numpy as np
from scipy.linalg import expm
rng = np.random.default_rng(int(4))
for N in (2, 3):
    H = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); H = H - H.conj().T; M = H - np.trace(H) / N * np.eye(N)
    rows = []
    for t in (1e-1, 5e-2, 2.5e-2):
        lhs = 1 - np.trace(expm(t * M)).real / N
        quad = -np.trace(t * M @ (t * M)).real / (2 * N)
        rows.append(abs(lhs - quad) / quad)
    print('SU(%d): relative error of the quadratic term at t = 0.1, 0.05, 0.025:' % N, ['%.2e' % r for r in rows],
          '; ratios %.2f %.2f (4 means O(t^2))' % (rows[0] / rows[1], rows[1] / rows[2]))
    print('   Re Tr M^3 = %.1e (zero: anti-Hermitian)' % np.trace(M @ M @ M).real)
