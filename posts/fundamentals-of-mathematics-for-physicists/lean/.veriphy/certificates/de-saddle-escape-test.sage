# name: de-saddle-escape-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.saddle_escape file "de-saddle-escape-test.sage"
# description: Sage's limit of |ceᵗ| at ∞ is +∞ for c = 1 (edge), -2, 1/7
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=B(limit(abs((1)*exp(x)), x=oo) == oo),
                    rhs=1, relation='==', edge=True),
    oracle_instance('-2', '_',
                    lhs=B(limit(abs((-2)*exp(x)), x=oo) == oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/7', '_',
                    lhs=B(limit(abs((1/7)*exp(x)), x=oo) == oo),
                    rhs=1, relation='==', edge=False),
]
