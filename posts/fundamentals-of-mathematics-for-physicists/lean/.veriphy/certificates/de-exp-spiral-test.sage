# name: de-exp-spiral-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_spiral file "de-exp-spiral-test.sage"
# description: e^{t(α + B)} = e^{αt}[cos βt + (sin βt/β)B] for B² = -β² by Sage: (B, α, β) = (J, 0, 1), (A2 + 1, -1, 1), (2J, 3, 2), at t = 1 (and t = 0, edge)
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
    oracle_instance('2', '!![0, 1; -1, 0]', '0', '1', '_', '_', '0',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=0), identity_matrix(SR,2)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '!![1, 1; -2, -1]', '-1', '1', '_', '_', '1',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1), (exp(-1)*(cos(1)*identity_matrix(SR,2) + sin(1)*matrix(SR,[[1,1],[-2,-1]]))).apply_map(lambda e: e.simplify_full())),
                    rhs=1, relation='==', edge=False),
    oracle_instance('2', '!![0, 2; -2, 0]', '3', '2', '_', '_', '1',
                    lhs=Z(((t_)*matrix(SR, [[3,2],[-2,3]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1), (exp(3)*(cos(2)*identity_matrix(SR,2) + sin(2)/2*matrix(SR,[[0,2],[-2,0]]))).apply_map(lambda e: e.simplify_full())),
                    rhs=1, relation='==', edge=False),
]
