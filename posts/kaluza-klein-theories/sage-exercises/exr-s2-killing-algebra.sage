# Exercise exr-s2-killing-algebra: the Killing vectors of the round sphere in the conventions of Part I,
# their brackets [K_a, K_b] = -eps_abc K_c (so f_ab^c = -eps_abc) and their inner products K_a . K_b = delta_ab - n_a n_b,
# with n the unit position vector of the point.
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
def bracket(U, V): return vector([sum(U[k] * diff(V[m], [th, ph][k]) - V[k] * diff(U[m], [th, ph][k]) for k in range(2)) for m in range(2)])
print("[K_1, K_2] =", bracket(K[0], K[1]).apply_map(lambda e: e.simplify_full()), " and -K_3 =", -K[2])
print("[PASS] [K_a, K_b] = -eps_abc K_c:", all((bracket(K[a], K[b]) + sum(lc(a, b, c) * K[c] for c in range(3))).apply_map(lambda e: e.simplify_full()).is_zero() for a in range(3) for b in range(3)))
h = diagonal_matrix([1, sin(th)^2])
G = matrix(SR, 3, 3, lambda a, b: (K[a] * h * K[b]).simplify_full())
print("K_a . K_b ="); print(G)
print("[PASS] K_a . K_b = delta_ab - n_a n_b:", (G - (identity_matrix(3) - n.column() * n.row())).apply_map(lambda e: e.simplify_full()).is_zero())
print("[PASS] the Killing vectors are tangent to the sphere and the matrix K_a . K_b has rank 2 (one combination, n^a K_a, vanishes at each point):", G.rank() == 2 and (sum(n[a] * K[a] for a in range(3))).apply_map(lambda e: e.simplify_full()).is_zero())
