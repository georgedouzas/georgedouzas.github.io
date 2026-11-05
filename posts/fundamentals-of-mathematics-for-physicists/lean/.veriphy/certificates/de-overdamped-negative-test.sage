# name: de-overdamped-negative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HarmonicOscillator.overdamped_negative file "de-overdamped-negative-test.sage"
# description: -γ + √(γ² - ω₀²) < 0 in interval arithmetic at (γ, ω₀) = (2, 1), (5/4, 1), (10, 1/10) (edge, weak damping ratio)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '1', '_', '_',
                    lhs=RIF(-(2) + sqrt((2)^2 - (1)^2)),
                    rhs=RIF(0), relation='<', edge=False),
    oracle_instance('5/4', '1', '_', '_',
                    lhs=RIF(-(5/4) + sqrt((5/4)^2 - (1)^2)),
                    rhs=RIF(0), relation='<', edge=False),
    oracle_instance('10', '1/10', '_', '_',
                    lhs=RIF(-(10) + sqrt((10)^2 - (1/10)^2)),
                    rhs=RIF(0), relation='<', edge=True),
]
