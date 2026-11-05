# name: de-kg-velocities-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.kg_velocities_bounds file "de-kg-velocities-test.sage"
# description: phase velocity > 1 > group velocity in interval arithmetic at (k, m) = (1, 1) (edge), (3, 1/2), (1/10, 2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '_', '_',
                    lhs=B(RIF(sqrt((1)^2+(1)^2)/(1)) > 1 and RIF((1)/sqrt((1)^2+(1)^2)) < 1),
                    rhs=1, relation='==', edge=True),
    oracle_instance('3', '1/2', '_', '_',
                    lhs=B(RIF(sqrt((3)^2+(1/2)^2)/(3)) > 1 and RIF((3)/sqrt((3)^2+(1/2)^2)) < 1),
                    rhs=1, relation='==', edge=False),
    oracle_instance('1/10', '2', '_', '_',
                    lhs=B(RIF(sqrt((1/10)^2+(2)^2)/(1/10)) > 1 and RIF((1/10)/sqrt((1/10)^2+(2)^2)) < 1),
                    rhs=1, relation='==', edge=False),
]
