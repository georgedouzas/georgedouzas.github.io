# name: de-eigen-det-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Eigenvalues.eigenvalue_iff_det file "de-eigen-det-test.sage"
# description: λ is an eigenvalue iff det(λ - A) = 0: Sage's eigenvalues and charpoly agree for A2 at λ = 0 (not an eigenvalue), tanks at λ = -2, 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_')
A2 = matrix(QQ, [[0, 1], [-2, -2]])
J = matrix(QQ, [[0, 1], [-1, 0]])
N = matrix(QQ, [[0, 1], [0, 0]])
instances = [
    oracle_instance('2', '!![0, 1; -2, -2]', '0',
                    lhs=B(0 in A2.eigenvalues()),
                    rhs=B((A2.charpoly())(0) == 0), relation='==', edge=False),
    oracle_instance('2', '!![-1, 1; 1, -1]', '-2',
                    lhs=B(-2 in matrix(QQ,[[-1,1],[1,-1]]).eigenvalues()),
                    rhs=B(matrix(QQ,[[-1,1],[1,-1]]).charpoly()(-2) == 0), relation='==', edge=False),
    oracle_instance('2', '0', '0',
                    lhs=B(0 in zero_matrix(QQ,2).eigenvalues()),
                    rhs=B(zero_matrix(QQ,2).charpoly()(0) == 0), relation='==', edge=True),
]
