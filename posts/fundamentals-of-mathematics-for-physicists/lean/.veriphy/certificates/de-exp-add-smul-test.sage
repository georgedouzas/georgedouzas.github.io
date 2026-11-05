# name: de-exp-add-smul-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.MatrixExponential.exp_add_smul file "de-exp-add-smul-test.sage"
# description: e^{(t+s)A} = e^{tA}e^{sA} by Sage at (t, s) = (0, 0) (edge), (1, 2), (-1/2, 3/2) for A = A2
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
    oracle_instance('2', '!![0, 1; -2, -2]', '0', '0',
                    lhs=((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=(0)+(0)),
                    rhs=(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=0)*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=0)).apply_map(lambda e: e.simplify_full()), relation='==', edge=True),
    oracle_instance('2', '!![0, 1; -2, -2]', '1', '2',
                    lhs=((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=(1)+(2)),
                    rhs=(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1)*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=2)).apply_map(lambda e: e.simplify_full()), relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]', '-1/2', '3/2',
                    lhs=((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=(-1/2)+(3/2)),
                    rhs=(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=-1/2)*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=3/2)).apply_map(lambda e: e.simplify_full()), relation='==', edge=False),
]
