# Exercise exr-fld-clifford-5d: five-dimensional gamma matrices for eta = diag(1, -1, -1, -1, -1): the four Weyl
# matrices plus gamma^4 = i gamma_5 satisfy {Gamma^M, Gamma^N} = 2 eta^{MN}; the product Gamma^0...Gamma^4 is a
# multiple of the identity, so there is no chirality in five dimensions.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
g5 = I * g[0] * g[1] * g[2] * g[3]
G = g + [I * g5]
eta5 = diagonal_matrix(SR, [1, -1, -1, -1, -1]); E4 = identity_matrix(SR, 4)
print("{Gamma^M, Gamma^N} = 2 eta^{MN} in five dimensions:", all(G[a] * G[b] + G[b] * G[a] == 2 * eta5[a, b] * E4 for a in range(5) for b in range(5)))
print("(Gamma^4)^2 =", (G[4]**2).diagonal(), "; the choice gamma^4 = gamma_5 fails:", (g5**2).diagonal())
P = G[0] * G[1] * G[2] * G[3] * G[4]
print("Gamma^0 Gamma^1 Gamma^2 Gamma^3 Gamma^4 =", P.diagonal(), "times the identity:", P == P[0, 0] * E4)
