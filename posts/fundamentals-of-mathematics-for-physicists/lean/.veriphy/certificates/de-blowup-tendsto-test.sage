# name: de-blowup-tendsto-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_tendsto file "de-blowup-tendsto-test.sage"
# description: Sage's one-sided limit of y₀/(1 - y₀x) at 1/y₀ from the left is +∞, for y₀ = 1 (edge), 2, 1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '_',
                    lhs=B(limit(1/(1-x), x=1, dir='-') == oo),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=B(limit(2/(1-2*x), x=1/2, dir='-') == oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/3', '_',
                    lhs=B(limit((1/3)/(1-x/3), x=3, dir='-') == oo),
                    rhs=1, relation='==', edge=False),
]
