# name: de-packet-modulus-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PDEComputations.packet_modulus file "de-packet-modulus-test.sage"
# description: |e^{iφ}w| = |w| by Sage at (φ, w) = (0, 0) (edge), (1, 3 + 4i), (π/3, -2i)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=SR(abs(exp(I*(0))*(0))).simplify_full(),
                    rhs=SR(abs(0)), relation='==', edge=True),
    oracle_instance('1', '3 + 4*I',
                    lhs=SR(abs(exp(I*(1))*(3 + 4*I))).simplify_full(),
                    rhs=SR(abs(3 + 4*I)), relation='==', edge=False),
    oracle_instance('pi/3', '-2*I',
                    lhs=SR(abs(exp(I*(pi/3))*(-2*I))).simplify_full(),
                    rhs=SR(abs(-2*I)), relation='==', edge=False),
]
