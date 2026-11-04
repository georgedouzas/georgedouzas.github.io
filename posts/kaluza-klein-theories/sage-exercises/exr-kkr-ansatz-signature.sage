# Exercise exr-kkr-ansatz-signature: the signature of the Kaluza-Klein ansatz. With the frame matrix E whose rows are
# the one-forms dx^mu and e^y = dy + A_mu dx^mu, the ansatz is E^T diag(g, -phi^2) E; E is invertible (det E = 1), so
# by Sylvester's law of inertia the ansatz has the signature of diag(g, -phi^2), i.e. (+,-,-,-,-) for g of signature
# (+,-,-,-). Checked here with a generic symmetric g and a numerical example.
gs = [[var("g%d%d" % (min(i, j), max(i, j))) for j in range(4)] for i in range(4)]
A = [var("A%d" % i) for i in range(4)]; phi = var("phi")
g = matrix(SR, gs)
G = matrix(SR, 5, 5)
for i in range(4):
    for j in range(4): G[i, j] = g[i, j] - phi^2 * A[i] * A[j]
    G[i, 4] = G[4, i] = -phi^2 * A[i]
G[4, 4] = -phi^2
E = identity_matrix(SR, 5)
for i in range(4): E[4, i] = A[i]                                   # e^y = dy + A_mu dx^mu
D = block_diagonal_matrix(g, matrix(SR, [[-phi^2]]))
print("[PASS] the ansatz is E^T diag(g, -phi^2) E with det E = 1:", bool((E.transpose() * D * E - G).simplify_full() == 0) and bool(E.det() == 1))
num = G.subs({gs[i][j]: (1 if i == j == 0 else (-1 if i == j else 0)) for i in range(4) for j in range(4)}).subs(A0=2, A1=-1, A2=3, A3=1/2, phi=3/2)
ev = num.change_ring(RDF).eigenvalues()
print("eigenvalues of the ansatz for g = eta, A = (2, -1, 3, 1/2), phi = 3/2:", [round(e, 4) for e in sorted(ev)])
print("[PASS] one positive and four negative eigenvalues: signature (+,-,-,-,-):", sum(1 for e in ev if e > 0) == 1 and sum(1 for e in ev if e < 0) == 4)
