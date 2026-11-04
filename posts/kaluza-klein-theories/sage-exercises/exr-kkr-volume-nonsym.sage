# Exercise exr-kkr-volume-nonsym: the Lean statement of the volume factor holds for any 4 x 4 matrix g, symmetric or
# not: det of the block matrix [[g - phi^2 A A^T, -phi^2 A], [-phi^2 A^T, -phi^2]] is -phi^2 det g as a polynomial
# identity in the 16 + 4 + 1 entries.
phi = var("phi")
g = matrix(SR, 4, 4, lambda i, j: var("m%d%d" % (i, j)))            # sixteen independent entries
A = [var("A%d" % i) for i in range(4)]
G = matrix(SR, 5, 5)
for i in range(4):
    for j in range(4): G[i, j] = g[i, j] - phi^2 * A[i] * A[j]
    G[i, 4] = G[4, i] = -phi^2 * A[i]
G[4, 4] = -phi^2
P = (G.det() + phi^2 * g.det()).expand()
print("number of terms of det g:", len(g.det().expand().operands()))
print("[PASS] det G = -phi^2 det g for a generic non-symmetric 4 x 4 matrix g:", bool(P == 0))
