# name: de-delta-series-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.DeltaFunction.delta_series_against_sine file "de-delta-series-test.sage"
# description: integrating the delta series against sin mx leaves sin mξ, by Sage at (N, m, ξ) = (1, 1, 1) (edge), (4, 2, 1/2), (5, 5, 2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '_', '1',
                    lhs=sum(2/pi*sin(n*(1))*integrate(sin(n*x)*sin(1*x), x, 0, pi) for n in range(1, 1+1)),
                    rhs=sin(1*(1)), relation='==', edge=True),
    oracle_instance('4', '2', '_', '1/2',
                    lhs=sum(2/pi*sin(n*(1/2))*integrate(sin(n*x)*sin(2*x), x, 0, pi) for n in range(1, 4+1)),
                    rhs=sin(2*(1/2)), relation='==', edge=False),
    oracle_instance('5', '5', '_', '2',
                    lhs=sum(2/pi*sin(n*(2))*integrate(sin(n*x)*sin(5*x), x, 0, pi) for n in range(1, 5+1)),
                    rhs=sin(5*(2)), relation='==', edge=False),
]
