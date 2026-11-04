# Exercise exr-sr-cover-numeric: the covering map A -> Lambda(A) from SL(2,C) to SO+(1,3), tested on random elements.
import numpy as np
rng = np.random.default_rng(int(1))
s = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
eta = np.diag([1., -1, -1, -1])
def Lam(A):
    return np.array([[0.5*np.trace(s[m] @ A @ s[n] @ A.conj().T).real for n in range(4)] for m in range(4)])
def rand_sl2c():
    A = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)); return A/np.sqrt(np.linalg.det(A))
ok = True
for _ in range(200):
    A, B = rand_sl2c(), rand_sl2c(); LA = Lam(A)
    ok &= np.allclose(LA.T @ eta @ LA, eta) and np.isclose(np.linalg.det(LA), 1) and LA[0, 0] >= 1
    ok &= np.allclose(Lam(A @ B), LA @ Lam(B)) and np.allclose(Lam(-A), LA)
print("Lambda(A) in SO+(1,3), homomorphism and Lambda(-A) = Lambda(A) on 200 random pairs:", ok)
eta_ = 0.8; A = np.cosh(eta_/2)*np.eye(2) + np.sinh(eta_/2)*s[1]
print("Lambda(exp(0.8 sigma_1/2)) ="); print(np.round(Lam(A), 6))
print("cosh 0.8 = %.6f, sinh 0.8 = %.6f" % (np.cosh(0.8), np.sinh(0.8)))
