# name: de-heat-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.heat_series_solves file "de-heat-series-test.sage"
# description: a finite sine series of decaying modes solves u_t = κu_xx by Sage: N = 0 (edge), N = 3 with (1, -1, 2), κ = 1/2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_')
instances = [
    oracle_instance('0', '_', '1', '1', '1',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
    oracle_instance('3', '_', '1/2', 't_', 'x',
                    lhs=(diff(sum(c*exp(-1/2*n^2*t_)*sin(n*x) for n, c in ((0, 1), (1, -1), (2, 2))), t_) - 1/2*diff(sum(c*exp(-1/2*n^2*t_)*sin(n*x) for n, c in ((0, 1), (1, -1), (2, 2))), x, 2)).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2', '_', '3', 't_', 'x',
                    lhs=(diff(5*exp(-3*t_)*sin(x), t_) - 3*diff(5*exp(-3*t_)*sin(x), x, 2)).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
