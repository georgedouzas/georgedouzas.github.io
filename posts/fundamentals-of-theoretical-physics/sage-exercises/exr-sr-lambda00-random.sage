# Exercise exr-sr-lambda00-random: the sign of Lambda^0_0 is multiplicative on random Lorentz matrices in all four
# components.
import numpy as np
rng = np.random.default_rng(int(7))
eta = np.diag([1., -1, -1, -1])
def boost(v):
    b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v; B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def rot():
    q, r = np.linalg.qr(rng.normal(size=(3, 3))); q = q*np.sign(np.diag(r)); q *= np.linalg.det(q)
    R = np.eye(4); R[1:, 1:] = q; return R
D = [np.eye(4), np.diag([1., -1, -1, -1]), np.diag([-1., 1, 1, 1]), -np.eye(4)]
def rand_lorentz():
    v = rng.normal(size=3); v *= rng.uniform(0, 0.99)/np.linalg.norm(v)
    return boost(v) @ rot() @ D[rng.integers(4)]
ok, smallest = True, 1e9
for _ in range(1000):
    A, B = rand_lorentz(), rand_lorentz()
    assert np.allclose(A.T @ eta @ A, eta)
    C = A @ B
    ok &= np.sign(C[0, 0]) == np.sign(A[0, 0])*np.sign(B[0, 0])
    smallest = min(smallest, abs(C[0, 0]))
print("sign multiplicative in all 1000 cases:", ok, "  smallest |(Lambda1 Lambda2)^0_0| = %.6f" % smallest)
