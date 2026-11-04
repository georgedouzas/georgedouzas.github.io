# Exercise exr-nak-sphere-radius-r (integrative): the non-abelian ansatz on a sphere of radius r, gamma = r^2 (dth^2 +
# sin^2 th dph^2), h = -gamma. The Killing vectors are unchanged in coordinates; their inner products scale to
# r^2 (delta_ab - n_a n_b); the block inverse holds and det ghat = det g det h = det g r^4 sin^2 th; and the Lie
# derivative of the mixed components still gives delta A^c = d eps^c + eps_abc A^a eps^b.
th, ph, r = var("th ph r"); assume(r > 0)
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
gam = r^2 * diagonal_matrix(SR, [1, sin(th)^2]); h = -gam
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
def lie_metric(X, m):                       # (L_X m)_{ij} = X^k d_k m_ij + d_i X^k m_kj + d_j X^k m_ik
    Y = [th, ph]
    return matrix(SR, 2, 2, lambda i, j: sum(X[k] * diff(m[i, j], Y[k]) + diff(X[k], Y[i]) * m[k, j] + diff(X[k], Y[j]) * m[i, k] for k in range(2)))
print("[PASS] the K_a are Killing for every radius:", all(lie_metric(K[a], gam).apply_map(lambda e: e.simplify_full()).is_zero() for a in range(3)))
G = matrix(SR, 3, 3, lambda a, b: (K[a] * gam * K[b]).simplify_full())
print("[PASS] K_a . K_b = r^2 (delta_ab - n_a n_b):", (G - r^2 * (identity_matrix(3) - n.column() * n.row())).apply_map(lambda e: e.simplify_full()).is_zero())
g = matrix(SR, [[var("g00"), var("g01")], [var("g01"), var("g11")]])
A = matrix(SR, 3, 2, lambda a, mu: var("A%d%d" % (a, mu)))
B = matrix(SR, 2, 3, lambda m, a: K[a][m]) * A
Gh = block_matrix([[g + B.transpose() * h * B, B.transpose() * h], [h * B, h]])
gi, hi = g.inverse(), h.inverse()
Gi = block_matrix([[gi, -gi * B.transpose()], [-B * gi, hi + B * gi * B.transpose()]])
print("[PASS] the block inverse inverts ghat on the sphere of radius r:", (Gh * Gi).apply_map(lambda e: e.simplify_full()) == identity_matrix(4))
print("[PASS] det ghat = det g r^4 sin^2 th:", bool((Gh.det() - g.det() * r^4 * sin(th)^2).simplify_full() == 0))
eps = [var("e%d" % a) for a in range(3)]; de = [var("de%d" % a) for a in range(3)]; Am = [var("a%d" % a) for a in range(3)]
ghat = [sum(h[m, n] * K[a][n] * Am[a] for a in range(3) for n in range(2)) for m in range(2)]    # ghat_{mu m} at fixed mu
xi = [sum(eps[a] * K[a][m] for a in range(3)) for m in range(2)]; dxi = [sum(de[a] * K[a][m] for a in range(3)) for m in range(2)]
Y = [th, ph]
delta = [sum(xi[k] * diff(ghat[m], Y[k]) + diff(xi[k], Y[m]) * ghat[k] + dxi[k] * h[k, m] for k in range(2)) for m in range(2)]
dA = [de[c] + sum(lc(a, b, c) * Am[a] * eps[b] for a in range(3) for b in range(3)) for c in range(3)]
cand = [sum(h[m, n] * K[c][n] * dA[c] for c in range(3) for n in range(2)) for m in range(2)]
print("[PASS] L_xi ghat_{mu m} = h_{mn} K^n_c (d eps^c + eps_abc A^a eps^b) for every radius:", all(bool((delta[m] - cand[m]).simplify_full() == 0) for m in range(2)))
