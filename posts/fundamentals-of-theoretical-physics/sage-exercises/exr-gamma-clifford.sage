# Exercise exr-gamma-clifford: the Dirac gamma matrices in the Weyl (chiral) representation with Sage: the Clifford
# relations {gamma^mu, gamma^nu} = 2 eta^{mu nu} for eta = diag(1, -1, -1, -1), the chirality matrix gamma_5, and the
# Lorentz generators S^{mu nu} = [gamma^mu, gamma^nu]/4 closing into so(3,1).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
g0 = block_matrix([[Z, one], [one, Z]], subdivide=False)                          # (gamma^0)^2 = +1 in the signature (+, -, -, -)
g = [g0] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]   # (gamma^i)^2 = -1
eta = diagonal_matrix(SR, [1, -1, -1, -1])
print("Clifford relations {gamma^mu, gamma^nu} = 2 eta^{mu nu} with eta = diag(1, -1, -1, -1):", all(g[a] * g[b] + g[b] * g[a] == 2 * eta[a, b] * identity_matrix(SR, 4) for a in range(4) for b in range(4)))
g5 = I * g[0] * g[1] * g[2] * g[3]
print("gamma_5 = i gamma^0 gamma^1 gamma^2 gamma^3 =", g5.list(), "; gamma_5^2 = 1:", g5**2 == identity_matrix(SR, 4), "; anticommutes with all gamma^mu:", all(g5 * ga + ga * g5 == 0 for ga in g))
S = [[(g[a] * g[b] - g[b] * g[a]) / 4 for b in range(4)] for a in range(4)]
# Lorentz algebra: [S^{ab}, S^{cd}] = eta^{bc} S^{ad} - eta^{ac} S^{bd} - eta^{bd} S^{ac} + eta^{ad} S^{bc} (the same form in either signature)
ok = all((S[a][b] * S[c][d] - S[c][d] * S[a][b]) == eta[b, c] * S[a][d] - eta[a, c] * S[b][d] - eta[b, d] * S[a][c] + eta[a, d] * S[b][c] for a in range(4) for b in range(4) for c in range(4) for d in range(4))
print("the generators S^{mu nu} = [gamma^mu, gamma^nu]/4 satisfy the so(3,1) algebra:", ok)
print("S^{12} = -(i/2) diag(sigma_3, sigma_3):", S[1][2] == -I / 2 * block_matrix([[s[2], Z], [Z, s[2]]], subdivide=False), "; a rotation by 2 pi about z gives exp(2 pi S^{12}) = -1:", bool((2 * pi * S[1][2]).exp().apply_map(lambda e: e.rectform().simplify_full()) == -identity_matrix(SR, 4)))
