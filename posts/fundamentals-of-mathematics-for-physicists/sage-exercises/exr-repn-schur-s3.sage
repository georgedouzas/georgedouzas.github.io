# Exercise exr-repn-schur-s3: commutants of three representations of S_3: the triangle (irreducible), the
# permutation representation (1 + 2) and the regular representation (1 + 1' + 2 + 2).
S3 = SymmetricGroup(3)
def commutant_dim(mats, n):
    R = PolynomialRing(QQ, n * n, 'x'); X = matrix(R, n, n, R.gens())
    eqs = [e for A in mats for e in (X * A - A * X).list()]
    M = matrix(QQ, [[e.coefficient(v) for v in R.gens()] for e in eqs])
    return n * n - M.rank()
def P(g):
    return matrix(QQ, 3, 3, lambda i, j: 1 if g(j + 1) == i + 1 else 0)
B = matrix(QQ, [[1, -1, 0], [0, 1, -1]]).transpose()
tri = [B.solve_right(P(g) * B) for g in S3]
els = S3.list()
reg = [matrix(QQ, 6, 6, lambda i, j: 1 if els[j] * g == els[i] else 0) for g in S3]   # left translation x -> g x
print("triangle (irreducible): commutant dimension", commutant_dim(tri, 2))
print("permutation = 1 + 2: commutant dimension", commutant_dim([P(g) for g in S3], 3))
print("regular = 1 + 1' + 2 x 2: commutant dimension", commutant_dim(reg, 6), "= 1^2 + 1^2 + 2^2")
