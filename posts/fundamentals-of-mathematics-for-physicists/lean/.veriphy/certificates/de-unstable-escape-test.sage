# name: de-unstable-escape-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.unstable_escape file "de-unstable-escape-test.sage"
# description: Sage's limit of e^{λt}c at ∞ is +∞, for (λ, c) = (1, 1) (edge), (1/2, 3), (2, 1/5)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '_', '_',
                    lhs=B(limit(exp((1)*x)*(1), x=oo) == oo),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1/2', '3', '_', '_',
                    lhs=B(limit(exp((1/2)*x)*(3), x=oo) == oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('2', '1/5', '_', '_',
                    lhs=B(limit(exp((2)*x)*(1/5), x=oo) == oo),
                    rhs=1, relation='==', edge=False),
]
