# Exercise exr-qm-runge-lenz-classical: the classical Runge-Lenz vector of the Kepler problem.
m, k = var("m k", domain="positive")
X = var("X1 X2 X3"); P = var("P1 P2 P3")
x = vector(SR, X); p = vector(SR, P); r = sqrt(x * x)
H = p * p / (2 * m) - k / r
L = x.cross_product(p); A = p.cross_product(L) - m * k * x / r
pb = lambda f, g: sum(diff(f, X[i]) * diff(g, P[i]) - diff(f, P[i]) * diff(g, X[i]) for i in range(3))
print("{H, A_i} =", [pb(H, A[i]).simplify_full() for i in range(3)])
print("A.L =", (A * L).simplify_full(), "; A^2 - m^2 k^2 - 2 m H L^2 =", (A * A - m ** 2 * k ** 2 - 2 * m * H * (L * L)).simplify_full())
eps = lambda i, j, l: sign(Permutation([i + 1, j + 1, l + 1])) if len({i, j, l}) == 3 else 0
print("{L_i, A_j} = eps_ijk A_k for all i, j:", all(bool((pb(L[i], A[j]) - sum(eps(i, j, l) * A[l] for l in range(3))).simplify_full() == 0) for i in range(3) for j in range(3)))
