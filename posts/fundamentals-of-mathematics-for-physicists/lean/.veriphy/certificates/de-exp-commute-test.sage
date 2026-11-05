# name: de-exp-commute-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_add_of_commute file "de-exp-commute-test.sage"
# description: e^{A+B} = e^Ae^B for commuting A, B by Sage: (1, N), (J, 2J), (0, 0) (edge)
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
    oracle_instance('2', '1', '!![0, 1; 0, 0]', '_',
                    lhs=(identity_matrix(QQ,2) + N).exp(),
                    rhs=(identity_matrix(QQ,2).exp()*N.exp()), relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -1, 0]', '!![0, 2; -2, 0]', '_',
                    lhs=(3*matrix(SR,J)).exp().apply_map(lambda e: e.simplify_full()),
                    rhs=(matrix(SR,J).exp()*(2*matrix(SR,J)).exp()).apply_map(lambda e: e.simplify_full()), relation='==', edge=False),
    oracle_instance('2', '0', '0', '_',
                    lhs=zero_matrix(QQ,2).exp(),
                    rhs=identity_matrix(QQ,2), relation='==', edge=True),
]
