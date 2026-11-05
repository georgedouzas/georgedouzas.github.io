# name: de-exp-diagonal-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_diagonal_smul file "de-exp-diagonal-test.sage"
# description: e^{t diag(d)} = diag(e^{tdᵢ}) by Sage: d = (1, -1), (2, 0, -3), (0) (edge), at t = 1/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '![1, -1]', '1/2',
                    lhs=(1/2*diagonal_matrix(QQ,[1,-1])).exp(),
                    rhs=diagonal_matrix(SR,[exp(1/2), exp(-1/2)]), relation='==', edge=False),
    oracle_instance('3', '![2, 0, -3]', '1/2',
                    lhs=(1/2*diagonal_matrix(QQ,[2,0,-3])).exp(),
                    rhs=diagonal_matrix(SR,[exp(1), 1, exp(-3/2)]), relation='==', edge=False),
    oracle_instance('1', '![0]', '1/2',
                    lhs=matrix(SR,[[exp(1/2*0)]]),
                    rhs=identity_matrix(QQ,1), relation='==', edge=True),
]
