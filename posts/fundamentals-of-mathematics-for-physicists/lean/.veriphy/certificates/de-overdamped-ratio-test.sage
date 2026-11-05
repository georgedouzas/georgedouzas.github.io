# name: de-overdamped-ratio-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HarmonicOscillator.overdamped_example_ratio file "de-overdamped-ratio-test.sage"
# description: e^{-3t} < 1/100 in interval arithmetic at t = 1.6 + 10⁻⁶ (edge), 2, 10
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('8/5 + 1/1000000', '_',
                    lhs=RIF(exp(-3*(8/5 + 1/1000000))),
                    rhs=RIF(1/100), relation='<', edge=True),
    oracle_instance('2', '_',
                    lhs=RIF(exp(-3*(2))),
                    rhs=RIF(1/100), relation='<', edge=False),
    oracle_instance('10', '_',
                    lhs=RIF(exp(-3*(10))),
                    rhs=RIF(1/100), relation='<', edge=False),
]
