# name: de-zero-energy-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.zero_energy file "de-zero-energy-test.sage"
# description: w_t² + c²w_x² = 0 holds only at w_t = w_x = 0: the instances (c, 0, 0) for c = 1 (edge), 2, 1/3, with the sum positive at a perturbed point
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '0', '0', '_', '_',
                    lhs=B(0^2 + (1)^2*0^2 == 0 and RIF((1/10)^2 + (1)^2*(1/10)^2) > 0),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '0', '0', '_', '_',
                    lhs=B(0^2 + (2)^2*0^2 == 0 and RIF((1/10)^2 + (2)^2*(1/10)^2) > 0),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/3', '0', '0', '_', '_',
                    lhs=B(0^2 + (1/3)^2*0^2 == 0 and RIF((1/10)^2 + (1/3)^2*(1/10)^2) > 0),
                    rhs=1, relation='==', edge=False),
]
