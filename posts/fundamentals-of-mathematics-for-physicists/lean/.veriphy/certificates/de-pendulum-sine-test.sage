# name: de-pendulum-sine-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Exercises.pendulum_sine_error file "de-pendulum-sine-test.sage"
# description: θ - θ³/6 < sin θ ≤ θ in interval arithmetic at θ = 1/100 (edge, small angle), 1/2, 3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1/100', '_',
                    lhs=B(RIF((1/100) - (1/100)^3/6) < RIF(sin(1/100)) and RIF(sin(1/100)) <= RIF(1/100)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1/2', '_',
                    lhs=B(RIF((1/2) - (1/2)^3/6) < RIF(sin(1/2)) and RIF(sin(1/2)) <= RIF(1/2)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('3', '_',
                    lhs=B(RIF((3) - (3)^3/6) < RIF(sin(3)) and RIF(sin(3)) <= RIF(3)),
                    rhs=1, relation='==', edge=False),
]
