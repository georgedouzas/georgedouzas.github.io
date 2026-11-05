# name: de-critical-faster-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HarmonicOscillator.critical_faster file "de-critical-faster-test.sage"
# description: -γ < -γ + √(γ² - ω₀²) in interval arithmetic at (γ, ω₀) = (2, 1), (5/4, 1) and (101/100, 1) (edge, near critical)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', '1', '_', '_',
                    lhs=RIF(-(2)),
                    rhs=RIF(-(2) + sqrt((2)^2 - (1)^2)), relation='<', edge=False),
    oracle_instance('5/4', '1', '_', '_',
                    lhs=RIF(-(5/4)),
                    rhs=RIF(-(5/4) + sqrt((5/4)^2 - (1)^2)), relation='<', edge=False),
    oracle_instance('101/100', '1', '_', '_',
                    lhs=RIF(-(101/100)),
                    rhs=RIF(-(101/100) + sqrt((101/100)^2 - (1)^2)), relation='<', edge=True),
]
