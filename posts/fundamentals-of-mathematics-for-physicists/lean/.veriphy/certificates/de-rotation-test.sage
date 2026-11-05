# name: de-rotation-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.rotation_never_stops file "de-rotation-test.sage"
# description: for E > 1 the level curve has no point with ω = 0: E(θ, 0) = -cos θ ≤ 1, checked at θ = 0 (edge), π, 2, with ω ≠ 0 at E = 1.5 points
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('Real.pi', '1', '_',
                    lhs=RIF(1^2/2 - cos(pi)),
                    rhs=RIF(1), relation='>', edge=True),
    oracle_instance('0', '3', '_',
                    lhs=RIF(3^2/2 - cos(0)),
                    rhs=RIF(1), relation='>', edge=False),
    oracle_instance('2', '3/2', '_',
                    lhs=RIF((3/2)^2/2 - cos(2)),
                    rhs=RIF(1), relation='>', edge=False),
]
