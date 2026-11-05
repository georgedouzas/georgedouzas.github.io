# name: de-string-mode-energy-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Exercises.string_mode_energy file "de-string-mode-energy-test.sage"
# description: ½∫₀^π((b sin x sin t)² + (b cos x cos t)²)dx by Sage's integral at (b, t) = (0, 0) (edge), (1, 1), (3, 2/3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=1/2*integrate(((0)*sin(x)*sin(0))^2 + ((0)*cos(x)*cos(0))^2, x, 0, pi),
                    rhs=pi/4*(0)^2*(sin(0)^2 + cos(0)^2), relation='==', edge=True),
    oracle_instance('1', '1',
                    lhs=1/2*integrate(((1)*sin(x)*sin(1))^2 + ((1)*cos(x)*cos(1))^2, x, 0, pi),
                    rhs=pi/4*(1)^2*(sin(1)^2 + cos(1)^2), relation='==', edge=False),
    oracle_instance('3', '2/3',
                    lhs=1/2*integrate(((3)*sin(x)*sin(2/3))^2 + ((3)*cos(x)*cos(2/3))^2, x, 0, pi),
                    rhs=pi/4*(3)^2*(sin(2/3)^2 + cos(2/3)^2), relation='==', edge=False),
]
