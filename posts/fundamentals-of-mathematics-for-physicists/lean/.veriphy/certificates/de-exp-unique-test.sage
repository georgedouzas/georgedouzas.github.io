# name: de-exp-unique-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_eq_of_ode file "de-exp-unique-test.sage"
# description: a matrix function with Ṙ = AR and R(0) = 1 is Sage's e^{tA}: the rotation for J, (1, t; 0, 1) for N, the identity for A = 0 (edge), at t = 1
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
    oracle_instance('2', '!![0, 1; -1, 0]', 'fun t => !![Real.cos t, Real.sin t; -Real.sin t, Real.cos t]', '_', '_', '1',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1), matrix(SR, [[cos(1), sin(1)], [-sin(1), cos(1)]])),
                    rhs=1, relation='==', edge=False),
    oracle_instance('2', '!![0, 1; 0, 0]', 'fun t => !![1, t; 0, 1]', '_', '_', '1',
                    lhs=Z(((t_)*matrix(SR, [[0,1],[0,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1), matrix(SR, [[1, 1], [0, 1]])),
                    rhs=1, relation='==', edge=False),
    oracle_instance('2', '0', 'fun _ => 1', '_', '_', '1',
                    lhs=Z(((t_)*matrix(SR, [[0,0],[0,0]])).exp().apply_map(lambda e: e.simplify_full()), identity_matrix(SR, 2)),
                    rhs=1, relation='==', edge=True),
]
