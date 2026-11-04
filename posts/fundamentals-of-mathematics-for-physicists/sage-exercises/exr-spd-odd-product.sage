# Exercise exr-spd-odd-product: in odd dimension d = 2 nu + 1 the product of all gamma matrices is i^nu times 1 for the
# standard Euclidean matrices and -i^nu times 1 for their negatives: two inequivalent representations.
import numpy as np
from functools import reduce
s1 = np.array([[0, 1], [1, 0]], dtype=complex); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.array([[1, 0], [0, -1]], dtype=complex)
kron = lambda ms: reduce(np.kron, ms, np.eye(1))
def euclid(d):
    n = d // 2; G = []
    for k in range(n):
        for s in (s1, s2):
            G.append(kron([s3] * k + [s] + [np.eye(2)] * (n - k - 1)))
    if d % 2: G.append(kron([s3] * n))
    return G
for d in (3, 5, 7, 9):
    nu = (d - 1) // 2; G = euclid(d); N = G[0].shape[0]
    P = reduce(np.dot, G, np.eye(N)); Pm = reduce(np.dot, [-x for x in G], np.eye(N))
    c, cm = (1j) ** (-nu) * P[0, 0], (1j) ** (-nu) * Pm[0, 0]
    print("d = %d: i^-nu * product = %s * 1 (scalar: %s); for -gamma: %s" % (d, complex(c).real, np.allclose(P, P[0, 0] * np.eye(N)), complex(cm).real))
