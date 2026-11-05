# name: de-separate-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparationOfVariables.separate file "de-separate-test.sage"
# description: T'/(κT) = X''/X when T'X = κTX'', for the heat mode T = e^{-κk²t}, X = sin kx at sample points, and κ = 1, T = X = 1 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '1', '1', '1', '_', '_', '_', '_',
                    lhs=1/(1*1),
                    rhs=1/1, relation='==', edge=True),
    oracle_instance('2', 'exp(-2)', '-2*exp(-2)', 'sin(1)', '-sin(1)', '_', '_', '_', '_',
                    lhs=(-2*exp(-2))/(2*exp(-2)),
                    rhs=(-sin(1))/sin(1), relation='==', edge=False),
    oracle_instance('1/2', 'exp(-2)', '-2*exp(-2)', 'sin(2)', '-4*sin(2)', '_', '_', '_', '_',
                    lhs=(-2*exp(-2))/((1/2)*exp(-2)),
                    rhs=(-4*sin(2))/sin(2), relation='==', edge=False),
]
