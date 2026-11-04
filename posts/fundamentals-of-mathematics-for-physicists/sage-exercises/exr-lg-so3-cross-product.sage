# Exercise exr-lg-so3-cross-product: for n.L = n_a L_a, (n.L) v = n x v, and [n.L, m.L] = (n x m).L: the
# commutator of so(3) is the cross product of R^3.
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
n = vector(SR, var("n1 n2 n3")); m = vector(SR, var("m1 m2 m3")); v = vector(SR, var("v1 v2 v3"))
def dotL(u): return sum(u[a] * L[a] for a in range(3))
print("(n.L) v - n x v =", (dotL(n) * v - n.cross_product(v)).apply_map(expand))
print("[n.L, m.L] - (n x m).L =", (dotL(n) * dotL(m) - dotL(m) * dotL(n) - dotL(n.cross_product(m))).expand().list())
print("entries (L_a)_bc = -eps_abc:", all(L[a][b, c] == -Permutation([a+1, b+1, c+1]).sign() if len({a, b, c}) == 3 else L[a][b, c] == 0 for a in range(3) for b in range(3) for c in range(3)))
