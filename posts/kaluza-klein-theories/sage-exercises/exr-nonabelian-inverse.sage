# Exercise exr-nonabelian-inverse: the block inverse of the non-abelian ansatz G = [[g + B^T h B, B^T h], [h B, h]]
# with B = K A, checked on symbolic blocks (g 2x2 symmetric for brevity, h = -gamma minus the round sphere metric,
# signature (+,-,-,-,-,-), B = K A with the three Killing vectors and A^a_mu symbols); det h = sin^2 th since k = 2.
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
h = -diagonal_matrix(SR, [1, sin(th)^2])
g = matrix(SR, [[var("g00"), var("g01")], [var("g01"), var("g11")]])
A = matrix(SR, 3, 2, lambda a, mu: var("A%d%d" % (a, mu)))
Kmat = matrix(SR, 2, 3, lambda m, a: K[a][m])
B = Kmat * A                                                   # B^m_mu = K^m_a A^a_mu
G = block_matrix([[g + B.transpose() * h * B, B.transpose() * h], [h * B, h]])
gi = g.inverse(); hi = h.inverse()
Ginv = block_matrix([[gi, -gi * B.transpose()], [-B * gi, hi + B * gi * B.transpose()]])
print("[PASS] G G^{-1} = 1 for the block inverse of the non-abelian ansatz:", (G * Ginv).apply_map(lambda e: e.simplify_full()) == identity_matrix(4))
print("[PASS] det G = det g det h (the mixing B drops out of the volume element):", bool((G.det() - g.det() * h.det()).simplify_full() == 0))
