# Exercise exr-repn-s3-matrices: the permutation representation of S_3 on R^3. With the text's convention
# (g h = first h, then g) the matrix P(g) with P(g) e_i = e_{g(i)} satisfies P(gh) = P(g) P(h). Sage composes left to
# right, so the text's product g h is h*g in Sage.
S3 = SymmetricGroup(3)
def P(g):
    return matrix(QQ, 3, 3, lambda i, j: 1 if g(j + 1) == i + 1 else 0)
def mul(g, h):          # the text's product g h
    return h * g
print("P(g h) = P(g) P(h) for all 36 pairs:", all(P(mul(g, h)) == P(g) * P(h) for g in S3 for h in S3))
print("P(e) = 1:", P(S3.one()) == identity_matrix(QQ, 3))
for g in S3:
    print("  g = %-8s det P(g) = %2d  trace P(g) = %d  P(g) rows %s" % (g, P(g).det(), P(g).trace(), [list(r) for r in P(g).rows()]))
print("the transposed convention e_i -> e_{g^-1(i)} is also a representation:", all(P(mul(g, h)**-1).transpose() == P(g**-1).transpose() * P(h**-1).transpose() for g in S3 for h in S3))
