# name: de-exp-diagonalizable-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_diagonalizable file "de-exp-diagonalizable-test.sage"
# description: e^{tA} = Pe^{tD}P⁻¹ by Sage for the tanks matrix (P from its eigenvectors), (2, 1; 1, 2), and P = 1, D = 0 (edge), at t = 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '!![-1, 1; 1, -1]', '!![1, 1; 1, -1]', '![0, -2]', '_', '_', '1',
                    lhs=(matrix(QQ,[[-1,1],[1,-1]])).exp(),
                    rhs=(matrix(QQ,[[1,1],[1,-1]])*diagonal_matrix(SR,[1,exp(-2)])*matrix(QQ,[[1,1],[1,-1]]).inverse()), relation='==', edge=False),
    oracle_instance('2', '!![2, 1; 1, 2]', '!![1, 1; 1, -1]', '![3, 1]', '_', '_', '1',
                    lhs=(matrix(QQ,[[2,1],[1,2]])).exp(),
                    rhs=(matrix(QQ,[[1,1],[1,-1]])*diagonal_matrix(SR,[exp(3),exp(1)])*matrix(QQ,[[1,1],[1,-1]]).inverse()), relation='==', edge=False),
    oracle_instance('2', '0', '1', '![0, 0]', '_', '_', '1',
                    lhs=zero_matrix(QQ,2).exp(),
                    rhs=identity_matrix(QQ,2), relation='==', edge=True),
]
