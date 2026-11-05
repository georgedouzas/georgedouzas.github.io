# name: de-heat-boundary-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.heat_series_boundary file "de-heat-boundary-test.sage"
# description: the sine series vanishes at x = 0 and x = π by Sage: N = 0 (edge), 3, 5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '_', '1', '1',
                    lhs=B(sum(sin(n*0) for n in range(0)) == 0 and sum(sin(n*pi) for n in range(0)) == 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('3', '_', '1', '1',
                    lhs=B(sum(sin(n*0) for n in range(3)) == 0 and sum(sin(n*pi) for n in range(3)) == 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('5', '_', '1', '1',
                    lhs=B(sum(sin(n*0) for n in range(5)) == 0 and sum(sin(n*pi) for n in range(5)) == 0),
                    rhs=1, relation='==', edge=False),
]
