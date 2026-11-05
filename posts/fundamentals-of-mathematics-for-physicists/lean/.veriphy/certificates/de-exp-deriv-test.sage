# name: de-exp-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SolutionSpace.hasDerivAt_exp_smul file "de-exp-deriv-test.sage"
# description: d/dt e^{tA} = Ae^{tA} by Sage's exp of symbolic matrices: A = J, A2, 0 (edge)
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
    oracle_instance('2', '!![0, 1; -1, 0]', '1',
                    lhs=(diff(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()), t_) - J*((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full())).apply_map(lambda e: e.simplify_full()),
                    rhs=zero_matrix(SR, 2), relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]', '1',
                    lhs=(diff(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()), t_) - A2*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())).apply_map(lambda e: e.simplify_full()),
                    rhs=zero_matrix(SR, 2), relation='==', edge=False),
    oracle_instance('1', '0', '0',
                    lhs=zero_matrix(SR, 1),
                    rhs=zero_matrix(SR, 1), relation='==', edge=True),
]
