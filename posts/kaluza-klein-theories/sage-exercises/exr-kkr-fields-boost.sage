# Exercise exr-kkr-fields-boost: under a four-dimensional Lorentz boost at fixed y the components of the ansatz
# transform so that g stays eta, A_mu transforms as a covector, A' = Lambda^T A, and phi is unchanged.
b = var("b"); A = vector(SR, var("A0 A1 A2 A3")); phi = var("phi")
eta = diagonal_matrix(SR, [1, -1, -1, -1])
L = identity_matrix(SR, 4); L[0, 0] = L[1, 1] = cosh(b); L[0, 1] = L[1, 0] = sinh(b)   # dx^mu/dx'^nu for a boost along x
def ansatz(g, A, phi):
    G = matrix(SR, 5, 5)
    for i in range(4):
        for j in range(4): G[i, j] = g[i, j] - phi^2 * A[i] * A[j]
        G[i, 4] = G[4, i] = -phi^2 * A[i]
    G[4, 4] = -phi^2
    return G
J = block_diagonal_matrix(L, matrix(SR, [[1]]))
Gp = (J.transpose() * ansatz(eta, A, phi) * J).simplify_full()
Ap = L.transpose() * A
print("A' =", [a.simplify_full() for a in Ap])
print("[PASS] J^T G(eta, A, phi) J = G(eta, Lambda^T A, phi): g is invariant, A a covector, phi a scalar:",
      bool((Gp - ansatz(eta, Ap, phi)).simplify_full() == 0) and bool((L.transpose() * eta * L - eta).simplify_full() == 0))
