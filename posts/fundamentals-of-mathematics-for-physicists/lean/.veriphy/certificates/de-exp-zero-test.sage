# name: de-exp-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_zero_smul file "de-exp-zero-test.sage"
# description: e^{0A} = 1 by Sage's matrix exp: A = J, A2, 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '!![0, 1; -1, 0]',
                    lhs=(0*matrix(QQ, [[0,1],[-1,0]])).exp(),
                    rhs=identity_matrix(QQ, 2), relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]',
                    lhs=(0*matrix(QQ, [[0,1],[-2,-2]])).exp(),
                    rhs=identity_matrix(QQ, 2), relation='==', edge=False),
    oracle_instance('2', '0',
                    lhs=(0*matrix(QQ, [[0,0],[0,0]])).exp(),
                    rhs=identity_matrix(QQ, 2), relation='==', edge=True),
]
