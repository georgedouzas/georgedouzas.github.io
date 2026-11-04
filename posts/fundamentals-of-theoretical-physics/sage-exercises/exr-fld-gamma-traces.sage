# Exercise exr-fld-gamma-traces: trace identities of the Weyl gamma matrices, signature (+, -, -, -):
# tr gamma^mu = 0, tr gamma^mu gamma^nu = 4 eta^{mu nu}, tr of three gammas = 0, tr gamma_5 = 0 and
# tr gamma^mu gamma^nu gamma^rho gamma^sigma = 4(eta^{mn} eta^{rs} - eta^{mr} eta^{ns} + eta^{ms} eta^{nr}).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
eta = diagonal_matrix(SR, [1, -1, -1, -1]); R4 = range(4)
print("tr gamma^mu = 0:", all(G.trace() == 0 for G in g))
print("tr gamma^mu gamma^nu = 4 eta^{mu nu}:", all((g[a] * g[b]).trace() == 4 * eta[a, b] for a in R4 for b in R4))
print("trace of a product of three gammas = 0:", all((g[a] * g[b] * g[c]).trace() == 0 for a in R4 for b in R4 for c in R4))
print("four gammas:", all((g[a] * g[b] * g[c] * g[d]).trace() == 4 * (eta[a, b] * eta[c, d] - eta[a, c] * eta[b, d] + eta[a, d] * eta[b, c]) for a in R4 for b in R4 for c in R4 for d in R4))
g5 = I * g[0] * g[1] * g[2] * g[3]
print("tr gamma_5 = 0:", g5.trace() == 0, "; tr gamma_5 gamma^0 gamma^1 gamma^2 gamma^3 =", (g5 * g[0] * g[1] * g[2] * g[3]).trace())
