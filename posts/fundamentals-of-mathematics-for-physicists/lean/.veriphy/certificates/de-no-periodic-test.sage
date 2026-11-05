# name: de-no-periodic-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.no_periodic_orbit file "de-no-periodic-test.sage"
# description: ∫₀ᵀ γq̇² > 0 when q̇ is not identically zero: (γ, T, q̇) = (1, 2π, cos), (1/2, 1, t) (q̇(1/2) ≠ 0), (3, 1, 1) (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '2 * Real.pi', '1', 'Real.cos', '_', '_', '_', '_',
                    lhs=integrate(cos(x)^2, x, 0, 2*pi),
                    rhs=0, relation='>', edge=False),
    oracle_instance('1/2', '1', '1/2', 'id', '_', '_', '_', '_',
                    lhs=integrate(x^2/2, x, 0, 1),
                    rhs=0, relation='>', edge=False),
    oracle_instance('3', '1', '1/2', 'fun _ => 1', '_', '_', '_', '_',
                    lhs=integrate(SR(3), x, 0, 1),
                    rhs=0, relation='>', edge=True),
]
