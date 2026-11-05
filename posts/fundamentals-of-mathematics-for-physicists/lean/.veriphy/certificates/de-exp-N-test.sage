# name: de-exp-N-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_N file "de-exp-N-test.sage"
# description: e^{tN} = (1, t; 0, 1) by Sage at t = 0 (edge), 1, -5/2
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
    oracle_instance('0',
                    lhs=((0)*N).exp(),
                    rhs=matrix(QQ, [[1, 0], [0, 1]]), relation='==', edge=True),
    oracle_instance('1',
                    lhs=((1)*N).exp(),
                    rhs=matrix(QQ, [[1, 1], [0, 1]]), relation='==', edge=False),
    oracle_instance('-5/2',
                    lhs=((-5/2)*N).exp(),
                    rhs=matrix(QQ, [[1, -5/2], [0, 1]]), relation='==', edge=False),
]
