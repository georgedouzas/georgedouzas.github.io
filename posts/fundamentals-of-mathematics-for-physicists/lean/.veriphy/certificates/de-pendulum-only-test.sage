# name: de-pendulum-only-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.pendulum_equilibria_only file "de-pendulum-only-test.sage"
# description: Sage's solve of sin θ = 0 at θ = 0 (edge), π, -4π returns integer multiples of π
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '_',
                    lhs=B(QQ(0/pi) in ZZ),
                    rhs=1, relation='==', edge=True),
    oracle_instance('Real.pi', '_',
                    lhs=B((pi/pi) in ZZ and sin(pi) == 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('-4 * Real.pi', '_',
                    lhs=B((-4*pi/pi) in ZZ and sin(-4*pi) == 0),
                    rhs=1, relation='==', edge=False),
]
