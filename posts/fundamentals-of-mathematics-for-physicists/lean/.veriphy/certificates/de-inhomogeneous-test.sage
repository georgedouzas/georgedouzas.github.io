# name: de-inhomogeneous-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearSystems.inhomogeneous_difference file "de-inhomogeneous-test.sage"
# description: the difference of two solutions of ẋ = Ax + b solves ẋ = Ax: driven oscillator solutions differing by a free one, and b = 0 (edge)
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
    oracle_instance('2', '!![0, 1; -1, 0]', 'fun t => ![0, Real.cos (2 * t)]', '_', '_', '1', '_', '_',
                    lhs=(diff(vector([cos(t_), -sin(t_)]), t_) - J*vector([cos(t_), -sin(t_)])).subs(t_=1),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
    oracle_instance('1', '0', 'fun _ => 0', 'fun _ => ![1]', 'fun _ => ![1]', '0', '_', '_',
                    lhs=vector(SR, [0]),
                    rhs=vector(SR, [0]), relation='==', edge=True),
    oracle_instance('2', '!![0, 1; -2, -2]', 'fun t => ![0, 1]', '_', '_', '2', '_', '_',
                    lhs=(diff(((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1, 0]), t_) - A2*((t_)*matrix(SR, [[0,1],[-2,-2]])).exp().apply_map(lambda e: e.simplify_full())*vector([1,0])).apply_map(lambda e: e.simplify_full()).subs(t_=2),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
]
