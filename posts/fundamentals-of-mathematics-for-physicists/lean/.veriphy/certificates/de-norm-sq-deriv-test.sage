# name: de-norm-sq-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.norm_sq_deriv file "de-norm-sq-deriv-test.sage"
# description: d/dt|x|² = 2x·Ax along e^{tA}x₀: A = J (|x|² constant), A2, 0 (edge), at t = 1
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
    oracle_instance('2', '!![0, 1; -1, 0]', '_', '1', '_',
                    lhs=diff(cos(t_)^2 + sin(t_)^2, t_).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]', '_', '1', '_',
                    lhs=(diff((((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0])).dot_product(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0])), t_) - 2*(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0])).dot_product(A2*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0]))).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('1', '0', 'fun _ => ![1]', '0', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
