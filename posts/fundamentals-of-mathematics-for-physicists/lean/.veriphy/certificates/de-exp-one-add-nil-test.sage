# name: de-exp-one-add-nil-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_one_add_nilpotent file "de-exp-one-add-nil-test.sage"
# description: e^{t(1 + N)} = eᵗ(1 + tN) by Sage: N, N = 0 (edge), (0, 0, 1; …), at t = 1
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
    oracle_instance('2', '!![0, 1; 0, 0]', '_', '1',
                    lhs=(identity_matrix(QQ,2) + N).exp(),
                    rhs=exp(1)*(identity_matrix(QQ,2) + N), relation='==', edge=False),
    oracle_instance('2', '0', '_', '1',
                    lhs=identity_matrix(QQ,2).exp(),
                    rhs=exp(1)*identity_matrix(QQ,2), relation='==', edge=True),
    oracle_instance('3', '!![0, 0, 1; 0, 0, 0; 0, 0, 0]', '_', '1',
                    lhs=(identity_matrix(QQ,3) + matrix(QQ,[[0,0,1],[0,0,0],[0,0,0]])).exp(),
                    rhs=exp(1)*(identity_matrix(QQ,3) + matrix(QQ,[[0,0,1],[0,0,0],[0,0,0]])), relation='==', edge=False),
]
