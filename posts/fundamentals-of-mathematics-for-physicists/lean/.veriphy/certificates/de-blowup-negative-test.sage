# name: de-blowup-negative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_negative_tendsto file "de-blowup-negative-test.sage"
# description: Sage's limit of y₀/(1 - y₀x) as x → ∞ for y₀ = -1 (edge), -2, -1/5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('-1', '_',
                    lhs=limit(-1/(1+x), x=oo),
                    rhs=0, relation='==', edge=True),
    oracle_instance('-2', '_',
                    lhs=limit(-2/(1+2*x), x=oo),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1/5', '_',
                    lhs=limit((-1/5)/(1+x/5), x=oo),
                    rhs=0, relation='==', edge=False),
]
