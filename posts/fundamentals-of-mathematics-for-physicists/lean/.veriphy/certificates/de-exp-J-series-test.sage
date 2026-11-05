# name: de-exp-J-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearSystemComputations.exp_J_series file "de-exp-J-series-test.sage"
# description: e^{tJ} = cos t·1 + sin t·J by Sage at t = 0 (edge), 1, -2
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
    oracle_instance('0',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=0), cos(0)*identity_matrix(SR,2) + sin(0)*matrix(SR,J)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1), cos(1)*identity_matrix(SR,2) + sin(1)*matrix(SR,J)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('-2',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=-2), cos(-2)*identity_matrix(SR,2) + sin(-2)*matrix(SR,J)),
                    rhs=1, relation='==', edge=False),
]
