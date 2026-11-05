# name: de-rotation-parts-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Eigenvalues.rotation_real_parts file "de-rotation-parts-test.sage"
# description: e^{it}·i = -sin t + i cos t, exactly at t = 0 (edge), π/3, 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=SR(exp(I*(0))*I).rectform(),
                    rhs=SR(-sin(0) + cos(0)*I).rectform(), relation='==', edge=True),
    oracle_instance('pi/3',
                    lhs=SR(exp(I*(pi/3))*I).rectform(),
                    rhs=SR(-sin(pi/3) + cos(pi/3)*I).rectform(), relation='==', edge=False),
    oracle_instance('2',
                    lhs=SR(exp(I*(2))*I).rectform(),
                    rhs=SR(-sin(2) + cos(2)*I).rectform(), relation='==', edge=False),
]
