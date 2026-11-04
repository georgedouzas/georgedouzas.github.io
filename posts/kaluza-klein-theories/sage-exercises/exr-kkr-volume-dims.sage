# Exercise exr-kkr-volume-dims: the determinant of the ansatz in every dimension. For a generic symmetric d x d metric
# g, a d-vector A and phi, the (d+1) x (d+1) ansatz G = [[g - phi^2 A A^T, -phi^2 A], [-phi^2 A^T, -phi^2]] has
# det G = -phi^2 det g, for d = 1, 2, 3, 4: the A-dependence cancels in every dimension.
phi = var("phi")
ok = True
for d in range(1, 5):
    g = matrix(SR, d, d, lambda i, j: var("g%d%d" % (min(i, j), max(i, j))))
    A = [var("A%d" % i) for i in range(d)]
    G = matrix(SR, d + 1, d + 1)
    for i in range(d):
        for j in range(d): G[i, j] = g[i, j] - phi^2 * A[i] * A[j]
        G[i, d] = G[d, i] = -phi^2 * A[i]
    G[d, d] = -phi^2
    r = bool((G.det() + phi^2 * g.det()).expand() == 0)
    print("d = %d: det G + phi^2 det g = 0:" % d, r)
    ok = ok and r
print("[PASS] det G = -phi^2 det g for d = 1, 2, 3, 4:", ok)
