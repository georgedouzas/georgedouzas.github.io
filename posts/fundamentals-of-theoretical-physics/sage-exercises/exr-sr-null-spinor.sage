# Exercise exr-sr-null-spinor (Blennow-Ohlsson, Problem 1.35): the spinor of a null vector and its transformation under
# the boost A = exp(-theta sigma_3/2), tanh(theta) = 0.6.
import numpy as np
s = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
X = lambda x: sum(x[m]*s[m] for m in range(4))
spinor = lambda x: np.array([x[0] + x[3], x[1] + 1j*x[2]])/np.sqrt(x[0] + x[3])
x = np.array([5., 3, 0, 4]); u = spinor(x)
print("u u^dagger = X(x):", np.allclose(np.outer(u, u.conj()), X(x)))
th = np.arctanh(0.6); A = np.diag([np.exp(-th/2), np.exp(th/2)])
Lam = np.array([[0.5*np.trace(s[m] @ A @ s[n] @ A.conj().T).real for n in range(4)] for m in range(4)])
xp = Lam @ x
print("Lambda(A) x =", np.round(xp, 6), "  (passive boost along z with v = 0.6: t' =", 1.25*(5 - 0.6*4), ", z' =", 1.25*(4 - 0.6*5), ")")
print("A u equals the spinor of Lambda(A) x:", np.allclose(A @ u, spinor(xp)))
