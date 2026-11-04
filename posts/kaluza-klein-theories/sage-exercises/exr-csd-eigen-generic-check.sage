# Exercise exr-csd-eigen-generic-check: decompose a random 3 x 3 complex matrix into the eigenspaces of ad tau by the
# spectral projectors P_lambda = prod_{mu != lambda} (ad tau - mu)/(lambda - mu), and check the result.
set_random_seed(3)
tau = I / 3 * diagonal_matrix(QQbar, [1, 1, -2]); ad = lambda X: tau * X - X * tau
X = matrix(QQbar, 3, 3, [QQbar(ZZ.random_element(-5, 6)) + I * ZZ.random_element(-5, 6) for _ in range(9)])
def P(lam, X):
    out = X
    for mu in [0, I, -I]:
        if mu != lam: out = (ad(out) - mu * out) / (lam - mu)
    return out
parts = {lam: P(lam, X) for lam in (0, I, -I)}
for lam, Y in parts.items(): print("component of eigenvalue", lam, ":"); print(Y)
print("[PASS] the three components add up to X and each is an eigenvector: [tau, X_lambda] = lambda X_lambda:",
      sum(parts.values()) == X and all(ad(Y) == lam * Y for lam, Y in parts.items()))
print("[PASS] X_{+i} is the (1,3), (2,3) part, X_{-i} the (3,1), (3,2) part, X_0 the block-diagonal part:",
      all(parts[I][a, b] == (X[a, b] if (a, b) in [(0, 2), (1, 2)] else 0) for a in range(3) for b in range(3))
      and all(parts[-I][a, b] == (X[a, b] if (a, b) in [(2, 0), (2, 1)] else 0) for a in range(3) for b in range(3)))
# minimal polynomial: ad tau (ad tau - i)(ad tau + i) = 0 on all of gl(3)
E = lambda i, j: matrix(QQbar, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
print("[PASS] ad tau (ad tau - i)(ad tau + i) = 0 on every matrix unit (ad tau is diagonalisable with eigenvalues 0, +-i):",
      all((lambda Z: ad(ad(ad(Z))) + ad(Z))(E(a, b)).is_zero() for a in range(3) for b in range(3)))
