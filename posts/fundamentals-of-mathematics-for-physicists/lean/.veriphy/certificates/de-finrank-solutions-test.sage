# name: de-finrank-solutions-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SolutionSpace.finrank_solutions file "de-finrank-solutions-test.sage"
# description: the solution space has dimension n: the fundamental matrix e^{tA} at t = 0 has rank n, for A = J, A2 and n = 1 (edge)
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
                    lhs=((t_)*matrix(SR, [[0,1],[-1,0]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1).rank(),
                    rhs=2, relation='==', edge=False),
    oracle_instance('2', '!![0, 1; -2, -2]',
                    lhs=((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full()).subs(t_=1).rank(),
                    rhs=2, relation='==', edge=False),
    oracle_instance('1', '0',
                    lhs=identity_matrix(QQ, 1).rank(),
                    rhs=1, relation='==', edge=True),
]
