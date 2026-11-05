# name: de-eigen-solution-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Eigenvalues.eigen_solution file "de-eigen-solution-test.sage"
# description: e^{λt}v solves ẋ = Ax for an eigenpair: (diag(1, -1), e₁, 1), (A2 over the tanks matrix (1, 1) with λ = 0, edge), (J-free 1×1 (2), 2)
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
    oracle_instance('2', '!![1, 0; 0, -1]', '![1, 0]', '1', '_', '1',
                    lhs=(diff(exp(t_)*vector([1,0]), t_) - matrix(SR,[[1,0],[0,-1]])*exp(t_)*vector([1,0])).subs(t_=1),
                    rhs=vector(SR,[0,0]), relation='==', edge=False),
    oracle_instance('2', '!![-1, 1; 1, -1]', '![1, 1]', '0', '_', '0',
                    lhs=matrix(SR,[[-1,1],[1,-1]])*vector([1,1]),
                    rhs=vector(SR,[0,0]), relation='==', edge=True),
    oracle_instance('1', '!![2]', '![1]', '2', '_', '3',
                    lhs=(diff(exp(2*t_), t_) - 2*exp(2*t_)).subs(t_=3),
                    rhs=0, relation='==', edge=False),
]
