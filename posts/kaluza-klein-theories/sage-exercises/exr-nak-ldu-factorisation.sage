# Exercise exr-nak-ldu-factorisation: the block-triangular factorisation behind @prp-nonabelian-inverse,
# ghat = L^T diag(g, h) L with L = [[1, 0], [B, 1]], for symbolic blocks (g 3x3 symmetric, h 2x2 symmetric, B 2x3);
# it gives det ghat = det g det h and ghat^{-1} = L^{-1} diag(g^{-1}, h^{-1}) L^{-T} with L^{-1} = [[1, 0], [-B, 1]].
p, k = 3, 2
g = matrix(SR, p, p, lambda i, j: var("g%d%d" % (min(i, j), max(i, j))))
h = matrix(SR, k, k, lambda i, j: var("h%d%d" % (min(i, j), max(i, j))))
B = matrix(SR, k, p, lambda m, mu: var("B%d%d" % (m, mu)))
G = block_matrix([[g + B.transpose() * h * B, B.transpose() * h], [h * B, h]])
L = block_matrix([[identity_matrix(p), zero_matrix(p, k)], [B, identity_matrix(k)]])
D = block_diagonal_matrix([g, h])
print("[PASS] ghat = L^T diag(g, h) L:", (G - L.transpose() * D * L).apply_map(lambda t: t.expand()).is_zero())
print("[PASS] det L = 1, hence det ghat = det g det h:", L.det() == 1 and bool((G.det() - g.det() * h.det()).expand() == 0))
Li = block_matrix([[identity_matrix(p), zero_matrix(p, k)], [-B, identity_matrix(k)]])
print("[PASS] L^{-1} = [[1, 0], [-B, 1]]:", (L * Li).apply_map(lambda t: t.expand()) == identity_matrix(p + k))
Ginv = Li * block_diagonal_matrix([g.inverse(), h.inverse()]) * Li.transpose()
gi, hi = g.inverse(), h.inverse()
target = block_matrix([[gi, -gi * B.transpose()], [-B * gi, hi + B * gi * B.transpose()]])
print("[PASS] L^{-1} diag(g^{-1}, h^{-1}) L^{-T} is the block inverse of @prp-nonabelian-inverse:", (Ginv - target).apply_map(lambda t: t.simplify_rational()).is_zero())
print("[PASS] and it inverts ghat:", (G * Ginv).apply_map(lambda t: t.simplify_rational()) == identity_matrix(p + k))
