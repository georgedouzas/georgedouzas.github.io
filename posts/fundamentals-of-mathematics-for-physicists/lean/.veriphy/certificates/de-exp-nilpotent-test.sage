# name: de-exp-nilpotent-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_nilpotent file "de-exp-nilpotent-test.sage"
# description: e^{tN} = 1 + tN for N² = 0 by Sage: N, (0, 0, 1; 0, 0, 0; 0, 0, 0), 0 (edge), at t = 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_')
def nf(e):
    return SR(e).expand().simplify_full().rectform().simplify_full()
def Z(L, R):
    return ZZ(1) if all(nf(e) == 0 for e in (matrix(SR, L) - matrix(SR, R)).list()) else ZZ(0)
A2 = matrix(QQ, [[0, 1], [-2, -2]])
J = matrix(QQ, [[0, 1], [-1, 0]])
N = matrix(QQ, [[0, 1], [0, 0]])
instances = [
    oracle_instance('2', '!![0, 1; 0, 0]', '_', '2',
                    lhs=(2*N).exp(),
                    rhs=identity_matrix(QQ,2) + 2*N, relation='==', edge=False),
    oracle_instance('3', '!![0, 0, 1; 0, 0, 0; 0, 0, 0]', '_', '2',
                    lhs=(2*matrix(QQ,[[0,0,1],[0,0,0],[0,0,0]])).exp(),
                    rhs=identity_matrix(QQ,3) + 2*matrix(QQ,[[0,0,1],[0,0,0],[0,0,0]]), relation='==', edge=False),
    oracle_instance('2', '0', '_', '2',
                    lhs=zero_matrix(QQ,2).exp(),
                    rhs=identity_matrix(QQ,2), relation='==', edge=True),
]
