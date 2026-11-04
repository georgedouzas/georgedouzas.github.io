# Exercise exr-clm-poisson-lrl: Poisson brackets of the Laplace-Runge-Lenz vector with H, with L and with itself.
x, y, z, px, py, pz = var("x y z px py pz", domain="real")
m, k = var("m k", domain="positive")
qs, ps = [x, y, z], [px, py, pz]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(3))
X = vector(qs); P = vector(ps); r = sqrt(X * X)
Lv = X.cross_product(P); A = P.cross_product(Lv) - m * k * X / r
H = P * P / (2 * m) - k / r
eps = lambda a, b, c: (a - b) * (b - c) * (c - a) / 2
print("{A_i, H} = 0:", all(bool(PB(A[i], H).simplify_full() == 0) for i in range(3)))
print("{L_i, A_j} = eps_ijk A_k:", all(bool((PB(Lv[i], A[j]) - sum(eps(i, j, c) * A[c] for c in range(3))).simplify_full() == 0) for i in range(3) for j in range(3)))
print("{A_i, A_j} = -2 m H eps_ijk L_k:", all(bool((PB(A[i], A[j]) + 2 * m * H * sum(eps(i, j, c) * Lv[c] for c in range(3))).simplify_full() == 0) for i in range(3) for j in range(i + 1, 3)))
