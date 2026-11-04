# Exercise exr-lgt-st-su2-sum: a sum of SU(2) matrices is a multiple of an SU(2) matrix, k V with k = sqrt(det).
import numpy as np
rng = np.random.default_rng(int(61))
q = rng.normal(size=(6, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
mat = lambda a: a[0] * np.eye(2) + 1j * sum(a[i + 1] * sig[i] for i in range(3))
Sig = sum(mat(a) for a in q)
k = np.sqrt(np.linalg.det(Sig).real); V = Sig / k
print('det Sigma = %.6f (real, positive); k = %.6f = |sum of the quaternions| = %.6f' % (np.linalg.det(Sig).real, k, np.linalg.norm(q.sum(0))))
print('V unitary: %s, det V = %.6f' % (np.allclose(V @ V.conj().T, np.eye(2)), np.linalg.det(V).real))
U = mat(q[0])
print('(1/2) Tr(U Sigma) = %.6f = k a0(U V) = %.6f' % (np.trace(U @ Sig).real / 2, k * np.trace(U @ V).real / 2))
