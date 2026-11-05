# name: de-pendulum-energy-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.pendulum_energy_deriv file "de-pendulum-energy-deriv-test.sage"
# description: d/dt(ω²/2 - cos θ) along θ = sin t, ω = t² (and constants, edge) at t = 0, 1
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('Real.sin', 'fun t => t ^ 2', 'Real.cos 1', '2', '1', '_', '_',
                    lhs=diff((x^2)^2/2 - cos(sin(x)), x).subs(x=1),
                    rhs=1*2 + sin(sin(1))*cos(1), relation='==', edge=False),
    oracle_instance('Real.sin', 'fun t => t ^ 2', '1', '0', '0', '_', '_',
                    lhs=diff((x^2)^2/2 - cos(sin(x)), x).subs(x=0),
                    rhs=0*0 + sin(sin(0))*1, relation='==', edge=False),
    oracle_instance('fun _ => 1', 'fun _ => 2', '0', '0', '0', '_', '_',
                    lhs=diff(SR(2)^2/2 - cos(1), x),
                    rhs=0, relation='==', edge=True),
]
