# Exercise exr-kkr-inverse-dual-frame: raising the index of the one-form e^y = dy + A_mu dx^mu with the inverse
# ansatz gives -phi^{-2} d_y, and raising dx^mu gives g^{mu nu} h_nu with h_nu = d_nu - A_nu d_y the horizontal lift:
# the inverse metric is diag(g^{-1}, -phi^{-2}) in the frame (h_mu, d_y).
gi = [[var("gi%d%d" % (min(i, j), max(i, j))) for j in range(4)] for i in range(4)]   # g^{mu nu} as free symbols
A = [var("A%d" % i) for i in range(4)]; phi = var("phi")
Au = [sum(gi[m][k] * A[k] for k in range(4)) for m in range(4)]                        # A^mu
Gi = matrix(SR, 5, 5)
for i in range(4):
    for j in range(4): Gi[i, j] = gi[i][j]
    Gi[i, 4] = Gi[4, i] = -Au[i]
Gi[4, 4] = -phi^(-2) + sum(A[m] * Au[m] for m in range(4))
ey = vector(SR, A + [1])
up = (ey * Gi).apply_map(lambda e: e.expand())
print("e^y raised:", list(up))
print("[PASS] G^{AB} (e^y)_B = -phi^{-2} (d_y)^A:", all(bool(up[k] == 0) for k in range(4)) and bool(up[4] == -phi^(-2)))
ok = True
for m in range(4):
    dxm = vector(SR, [1 if k == m else 0 for k in range(5)])
    v = (dxm * Gi).apply_map(lambda e: e.expand())
    target = sum(gi[m][k] * vector(SR, [1 if l == k else 0 for l in range(4)] + [-A[k]]) for k in range(4))
    ok = ok and all(bool((v[l] - target[l]).expand() == 0) for l in range(5))
print("[PASS] G^{AB} (dx^mu)_B = g^{mu nu} h_nu with h_nu = d_nu - A_nu d_y:", ok)
