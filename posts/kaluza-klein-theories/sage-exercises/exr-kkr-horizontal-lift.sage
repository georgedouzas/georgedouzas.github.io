# Exercise exr-kkr-horizontal-lift: the horizontal lifts h_mu = d_mu - A_mu d_y of the coordinate vectors are
# orthogonal to the fibre direction d_y, and their inner products are those of the four-dimensional metric:
# G(h_mu, d_y) = 0, G(h_mu, h_nu) = g_mu nu, G(d_y, d_y) = -phi^2.
gs = [[var("g%d%d" % (min(i, j), max(i, j))) for j in range(4)] for i in range(4)]
A = [var("A%d" % i) for i in range(4)]; phi = var("phi")
G = matrix(SR, 5, 5)
for i in range(4):
    for j in range(4): G[i, j] = gs[i][j] - phi^2 * A[i] * A[j]
    G[i, 4] = G[4, i] = -phi^2 * A[i]
G[4, 4] = -phi^2
dy = vector(SR, [0, 0, 0, 0, 1])
h = [vector(SR, [1 if k == m else 0 for k in range(4)] + [-A[m]]) for m in range(4)]
print("[PASS] G(h_mu, d_y) = 0 for every mu:", all(bool((h[m] * G * dy).expand() == 0) for m in range(4)))
print("[PASS] G(h_mu, h_nu) = g_mu nu:", all(bool((h[m] * G * h[k] - gs[m][k]).expand() == 0) for m in range(4) for k in range(4)))
print("[PASS] G(d_y, d_y) = -phi^2:", bool(dy * G * dy == -phi^2))
