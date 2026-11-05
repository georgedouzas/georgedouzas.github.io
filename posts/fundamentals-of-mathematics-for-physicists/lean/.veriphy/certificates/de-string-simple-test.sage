# name: de-string-simple-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SturmLiouvilleTheorem.string_simple file "de-string-simple-test.sage"
# description: a solution of u'' = -ω²u with u(0) = 0 is (u'(0)/ω) sin ωx: Sage's desolve at (ω, u'(0)) = (1, 0) (edge), (2, 3), (1/2, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
U = function('U')
instances = [
    oracle_instance('1', '_', '_', '_', '_', '_', '_', '1',
                    lhs=desolve(diff(U(x),x,2) + (1)^2*U(x) == 0, U(x), ics=[0, 0, 0]),
                    rhs=(0)/(1)*sin((1)*x), relation='==', edge=True),
    oracle_instance('2', '_', '_', '_', '_', '_', '_', '1',
                    lhs=desolve(diff(U(x),x,2) + (2)^2*U(x) == 0, U(x), ics=[0, 0, 3]),
                    rhs=(3)/(2)*sin((2)*x), relation='==', edge=False),
    oracle_instance('1/2', '_', '_', '_', '_', '_', '_', '1',
                    lhs=desolve(diff(U(x),x,2) + (1/2)^2*U(x) == 0, U(x), ics=[0, 0, -1]),
                    rhs=(-1)/(1/2)*sin((1/2)*x), relation='==', edge=False),
]
