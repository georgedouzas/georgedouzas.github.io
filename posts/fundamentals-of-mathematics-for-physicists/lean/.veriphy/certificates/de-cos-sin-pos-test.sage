# name: de-cos-sin-pos-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.cos_sin_pos file "de-cos-sin-pos-test.sage"
# description: cos(sin x) > 0 in interval arithmetic at x = 0 (edge), π/2 (|sin| = 1), 4
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=RIF(cos(sin(0))),
                    rhs=RIF(0), relation='>', edge=True),
    oracle_instance('Real.pi / 2',
                    lhs=RIF(cos(sin(pi/2))),
                    rhs=RIF(0), relation='>', edge=True),
    oracle_instance('4',
                    lhs=RIF(cos(sin(4))),
                    rhs=RIF(0), relation='>', edge=False),
]
