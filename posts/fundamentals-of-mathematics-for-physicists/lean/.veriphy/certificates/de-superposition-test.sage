# name: de-superposition-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LinearSystems.superposition file "de-superposition-test.sage"
# description: αx + βy of solutions of ẋ = Ax solves it: A = J with the rotation solutions, A = 0 (edge), A = diag(1, -1), at t = 1
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
    oracle_instance('2', '!![0, 1; -1, 0]', 'fun t => ![Real.cos t, -Real.sin t]', 'fun t => ![Real.sin t, Real.cos t]', '2', '-3', '1', '_', '_',
                    lhs=(diff(vector([2*cos(t_) - 3*sin(t_), -2*sin(t_) - 3*cos(t_)]), t_) - J*vector([2*cos(t_) - 3*sin(t_), -2*sin(t_) - 3*cos(t_)])).subs(t_=1),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
    oracle_instance('1', '0', 'fun _ => ![1]', 'fun _ => ![2]', '0', '0', '0', '_', '_',
                    lhs=vector(SR, [0]),
                    rhs=vector(SR, [0]), relation='==', edge=True),
    oracle_instance('2', '!![1, 0; 0, -1]', 'fun t => ![Real.exp t, 0]', 'fun t => ![0, Real.exp (-t)]', '1', '1', '2', '_', '_',
                    lhs=(diff(vector([exp(t_), exp(-t_)]), t_) - matrix(SR, [[1,0],[0,-1]])*vector([exp(t_), exp(-t_)])).subs(t_=2),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
]
