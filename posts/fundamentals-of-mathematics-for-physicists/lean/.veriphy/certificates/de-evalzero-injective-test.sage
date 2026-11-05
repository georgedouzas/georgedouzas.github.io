# name: de-evalzero-injective-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SolutionSpace.evalZero_injective file "de-evalzero-injective-test.sage"
# description: two solutions of ẋ = Ax with the same x(0) coincide: Sage's desolve_system returns one solution, at A = J, A2, 0 (edge)
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
    oracle_instance('2', '!![0, 1; -1, 0]',
                    lhs=(((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full())*vector([1, 0])).subs(t_=0),
                    rhs=vector(SR, [1, 0]), relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]',
                    lhs=(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([0, 1])).subs(t_=0),
                    rhs=vector(SR, [0, 1]), relation='==', edge=False),
    oracle_instance('1', '0',
                    lhs=vector(SR, [1]),
                    rhs=vector(SR, [1]), relation='==', edge=True),
]
