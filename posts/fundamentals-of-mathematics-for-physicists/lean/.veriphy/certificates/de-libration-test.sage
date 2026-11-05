# name: de-libration-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.libration_bounded file "de-libration-test.sage"
# description: for E < 1, cos θ > -1: at (θ, ω) = (0, 0) (edge), (1, 1/2), (-2, 0)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0', '_',
                    lhs=RIF(cos(0)),
                    rhs=RIF(-1), relation='>', edge=True),
    oracle_instance('1', '1/2', '_',
                    lhs=RIF(cos(1)),
                    rhs=RIF(-1), relation='>', edge=False),
    oracle_instance('-2', '0', '_',
                    lhs=RIF(cos(-2)),
                    rhs=RIF(-1), relation='>', edge=False),
]
