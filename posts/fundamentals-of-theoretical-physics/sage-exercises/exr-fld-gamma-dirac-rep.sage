# Exercise exr-fld-gamma-dirac-rep: the Dirac representation gamma_D^0 = diag(1, -1) (in 2x2 blocks),
# gamma_D^k = [[0, sigma_k], [-sigma_k, 0]] satisfies the Clifford relations with eta = diag(1, -1, -1, -1), and is
# related to the Weyl matrices by the orthogonal U = [[1, 1], [-1, 1]]/sqrt(2): gamma_D^mu = U gamma_W^mu U^T.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
gW = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
gD = [block_matrix([[one, Z], [Z, -one]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
eta = diagonal_matrix(SR, [1, -1, -1, -1]); E4 = identity_matrix(SR, 4)
print("Dirac matrices satisfy {gamma^mu, gamma^nu} = 2 eta^{mu nu}:", all(gD[a] * gD[b] + gD[b] * gD[a] == 2 * eta[a, b] * E4 for a in range(4) for b in range(4)))
U = block_matrix([[one, one], [-one, one]], subdivide=False) / sqrt(2)
print("U is orthogonal:", U * U.transpose() == E4, "; U gamma_W^mu U^T = gamma_D^mu:", all((U * gW[a] * U.transpose()).simplify_full() == gD[a] for a in range(4)))
g5D = (I * gD[0] * gD[1] * gD[2] * gD[3])
print("gamma_5 in the Dirac representation:", g5D.list(), "(off-diagonal: chirality is not diagonal there)")
