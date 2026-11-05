# name: de-energy-zero-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.energy_stays_zero file "de-energy-zero-test.sage"
# description: a non-negative non-increasing energy starting at 0 stays 0: E = 0 (edge), and the heat energy of the zero solution at t = 1, 5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun _ => 0', '_', '_', '_', '0', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
    oracle_instance('fun _ => 0', '_', '_', '_', '1', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun _ => 0', '_', '_', '_', '5', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=False),
]
