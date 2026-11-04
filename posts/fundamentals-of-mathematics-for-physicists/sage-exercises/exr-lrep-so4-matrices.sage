# Exercise exr-lrep-so4-matrices: the rotation generators (M_ij)_kl = delta_ik delta_jl - delta_il delta_jk of R^4 (axes
# 0..3), the combinations J_a = -1/2 (M_bc + M_a3), K_a = -1/2 (M_bc - M_a3) for (a, b, c) cyclic in (0, 1, 2), their
# brackets, and the eigenvalues of (i J_2, i K_2) on C^4: the weights (+-1/2, +-1/2) of the vector representation.
def M(i, j):
    return matrix(QQ, 4, 4, lambda k, l: (1 if (k, l) == (i, j) else 0) - (1 if (k, l) == (j, i) else 0))
cyc = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
J = [-(M(b, c) + M(a, 3)) / 2 for a, b, c in cyc]
K = [-(M(b, c) - M(a, 3)) / 2 for a, b, c in cyc]
br = lambda A, B: A * B - B * A
print("[J_a, J_b] = J_c (cyclic):", all(br(J[a], J[b]) == J[c] for a, b, c in cyc))
print("[K_a, K_b] = K_c (cyclic):", all(br(K[a], K[b]) == K[c] for a, b, c in cyc))
print("[J_a, K_b] = 0 for all a, b:", all(br(X, Y) == 0 for X in J for Y in K))
print("J_0 = Lean's J 0:", J[0] == matrix(QQ, [[0,0,0,-1/2],[0,0,-1/2,0],[0,1/2,0,0],[1/2,0,0,0]]))
P = (I * J[2]).change_ring(QQbar); Q = (I * K[2]).change_ring(QQbar)
V = (QQbar^4)
ws = []
for ev, vecs, m in P.eigenvectors_right():
    sub = matrix(QQbar, vecs).transpose()
    Qr = sub.solve_right(Q * sub)                 # i K_2 restricted to the i J_2 eigenspace (they commute)
    ws += [(QQ(ev), QQ(e)) for e in Qr.eigenvalues()]
print("weights (i J_2, i K_2) of R^4 (x) C:", sorted(ws))
