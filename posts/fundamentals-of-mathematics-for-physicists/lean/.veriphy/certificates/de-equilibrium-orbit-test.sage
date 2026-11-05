# name: de-equilibrium-orbit-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.equilibrium_is_orbit file "de-equilibrium-orbit-test.sage"
# description: the constant at an equilibrium solves ẋ = F(x): Lotka-Volterra at (0, 0) (edge) and (1, 1), pendulum at (π, 0)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('2', 'fun v => ![v 0 - v 0 * v 1, -v 1 + v 0 * v 1]', '![0, 0]', '_', '0',
                    lhs=vector([0-0*0, -0+0*0]),
                    rhs=vector([0, 0]), relation='==', edge=True),
    oracle_instance('2', 'fun v => ![v 0 - v 0 * v 1, -v 1 + v 0 * v 1]', '![1, 1]', '_', '1',
                    lhs=vector([1-1*1, -1+1*1]),
                    rhs=vector([0, 0]), relation='==', edge=False),
    oracle_instance('2', 'fun v => ![v 1, -Real.sin (v 0)]', '![Real.pi, 0]', '_', '2',
                    lhs=vector(SR, [0, -sin(pi)]),
                    rhs=vector(SR, [0, 0]), relation='==', edge=False),
]
