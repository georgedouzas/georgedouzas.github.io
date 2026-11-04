# Exercise exr-schur-commutant: Schur's lemma for the vector representation of so(3): the commutant is the scalars,
# while for the reducible representation L_a (+) L_a (block-diagonal, 6 x 6) the commutant is 4-dimensional.
L = [matrix(QQ, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(QQ, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(QQ, [[0,-1,0],[1,0,0],[0,0,0]])]
def commutant_dim(gens, n):
    """Dimension of the space of n x n matrices X with X A = A X for every generator A: solve the linear system."""
    R = PolynomialRing(QQ, n*n, 'x'); X = matrix(R, n, n, R.gens())
    eqs = [e for A in gens for e in (X*A - A*X).list()]
    M = matrix(QQ, [[e.coefficient(v) for v in R.gens()] for e in eqs])
    return n*n - M.rank()
print("commutant of the vector representation (irreducible): dimension", commutant_dim(L, 3), " -> only multiples of the identity")
L2 = [block_diagonal_matrix(A, A) for A in L]
print("commutant of L (+) L (two copies of the same irreducible): dimension", commutant_dim(L2, 6), " = 2 x 2 = the matrices acting on the multiplicity space")
L3 = [block_diagonal_matrix(A, matrix(QQ, 1, 1, [0])) for A in L]       # vector (+) trivial
print("commutant of vector (+) trivial (two different irreducibles): dimension", commutant_dim(L3, 4), " = 1 + 1")
