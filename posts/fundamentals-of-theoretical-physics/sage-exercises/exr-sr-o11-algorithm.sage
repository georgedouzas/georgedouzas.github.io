# Exercise exr-sr-o11-algorithm: decomposition of an element of O(1,1) as L(eta) D with D in {1, P, T, PT}.
import numpy as np
L = lambda e: np.array([[np.cosh(e), np.sinh(e)], [np.sinh(e), np.cosh(e)]])
P = np.diag([1., -1]); T = np.diag([-1., 1])
names = {(1, 1): "1", (1, -1): "P", (-1, 1): "T", (-1, -1): "PT"}
def decompose(M):
    eps = np.sign(M[0, 0]); eta = np.arctanh(M[1, 0]/M[0, 0]); delta = np.sign(M[1, 1])
    D = np.diag([eps, delta])
    assert np.allclose(L(eta) @ D, M)
    return eta, names[(int(eps), int(delta))]
M = L(0.3) @ P @ L(-1.1) @ T @ L(0.5)
print("Lambda =", np.round(M, 6))
print("in O(1,1):", np.allclose(M.T @ np.diag([1., -1]) @ M, np.diag([1., -1])))
print("decomposition (eta, D):", decompose(M))
