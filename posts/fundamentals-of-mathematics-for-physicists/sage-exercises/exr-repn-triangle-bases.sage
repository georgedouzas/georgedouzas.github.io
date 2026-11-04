# Exercise exr-repn-triangle-bases: the triangle representation of S_3 on the plane x1 + x2 + x3 = 0 in two bases,
# B = (e1 - e2, e2 - e3) and B' = (e1 - e2, e1 + e2 - 2 e3); the change of basis S intertwines the two.
S3 = SymmetricGroup(3)
def P(g):
    return matrix(QQ, 3, 3, lambda i, j: 1 if g(j + 1) == i + 1 else 0)
B = matrix(QQ, [[1, -1, 0], [0, 1, -1]]).transpose()          # columns are the basis vectors
Bp = matrix(QQ, [[1, -1, 0], [1, 1, -2]]).transpose()
def rep(g, basis):
    # solve basis * M = P(g) * basis for the 2 x 2 matrix M
    return basis.solve_right(P(g) * basis)
S = Bp.solve_right(B)       # coordinates in B' of the vectors of B: v_B' = S v_B
print("S =", [list(r) for r in S.rows()])
print("S rho_B(g) S^-1 = rho_B'(g) for all g:", all(S * rep(g, B) * S.inverse() == rep(g, Bp) for g in S3))
for g in S3:
    print("  g = %-8s rho_B = %-22s rho_B' = %s" % (g, [list(r) for r in rep(g, B).rows()], [list(r) for r in rep(g, Bp).rows()]))
print("traces agree:", all(rep(g, B).trace() == rep(g, Bp).trace() for g in S3))
