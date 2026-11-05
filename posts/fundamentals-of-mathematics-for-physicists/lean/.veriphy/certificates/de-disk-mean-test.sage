# name: de-disk-mean-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LaplaceEquation.disk_mean file "de-disk-mean-test.sage"
# description: the mean over the circle is a₀/2 by Sage's integrals: N = 0 (edge), N = 2 and N = 3 with sample coefficients
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '3', '_', '_',
                    lhs=1/(2*pi)*integrate(SR(3)/2, x, 0, 2*pi),
                    rhs=3/2, relation='==', edge=True),
    oracle_instance('2', '1', '_', '_',
                    lhs=1/(2*pi)*integrate(1/2 + 2*cos(x) - sin(x) + 5*cos(2*x) + sin(2*x)/3, x, 0, 2*pi),
                    rhs=1/2, relation='==', edge=False),
    oracle_instance('3', '-4', '_', '_',
                    lhs=1/(2*pi)*integrate(-2 + cos(3*x) + 7*sin(2*x), x, 0, 2*pi),
                    rhs=-2, relation='==', edge=False),
]
