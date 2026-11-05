# name: de-exp-mulvec-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SolutionSpace.exp_mulVec_solves file "de-exp-mulvec-test.sage"
# description: x(t) = e^{tA}x₀ solves ẋ = Ax: (A, x₀) = (J, (1, 0)), (A2, (1, 1)), (0, (3)) (edge)
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
    oracle_instance('2', '!![0, 1; -1, 0]', '![1, 0]', '1',
                    lhs=(diff(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0]), t_) - J*((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0])).apply_map(lambda e: e.simplify_full()),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]', '![1, 1]', '1',
                    lhs=(diff(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,1]), t_) - A2*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,1])).apply_map(lambda e: e.simplify_full()),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
    oracle_instance('1', '0', '![3]', '0',
                    lhs=vector(SR, [0]),
                    rhs=vector(SR, [0]), relation='==', edge=True),
]
