# Exercise exr-qm-lz-commutator-sage: [L_i, L_j] = i hbar eps_ijk L_k and [L^2, L_z] = 0 on a symbolic f(x, y, z).
x, y, z, hbar = var("x y z hbar"); X = [x, y, z]
f = function("f")(x, y, z)
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1])) if len({i, j, k}) == 3 else 0
L = lambda i, g: -I * hbar * sum(eps(i, j, k) * X[j] * diff(g, X[k]) for j in range(3) for k in range(3))
for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    print("[L_%d, L_%d] f - i hbar L_%d f =" % (i + 1, j + 1, k + 1), (L(i, L(j, f)) - L(j, L(i, f)) - I * hbar * L(k, f)).simplify_full())
L2 = lambda g: sum(L(i, L(i, g)) for i in range(3))
print("[L^2, L_z] f =", (L2(L(2, f)) - L(2, L2(f))).simplify_full())
