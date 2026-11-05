# name: de-plucked-periodic-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.plucked_periodic file "de-plucked-periodic-test.sage"
# description: sin nx cos n(t + 2π) = sin nx cos nt by Sage at n = 0 (edge), 1, 5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_ = var('t_')
instances = [
    oracle_instance('0', 't_', 'x',
                    lhs=(sin(0*x)*cos(0*(t_ + 2*pi))).simplify_full(),
                    rhs=sin(0*x)*cos(0*t_), relation='==', edge=True),
    oracle_instance('1', 't_', 'x',
                    lhs=(sin(1*x)*cos(1*(t_ + 2*pi))).simplify_full(),
                    rhs=sin(1*x)*cos(1*t_), relation='==', edge=False),
    oracle_instance('5', 't_', 'x',
                    lhs=(sin(5*x)*cos(5*(t_ + 2*pi))).simplify_full(),
                    rhs=sin(5*x)*cos(5*t_), relation='==', edge=False),
]
