# name: de-mode-maximum-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.mode_maximum file "de-mode-maximum-test.sage"
# description: |e^{-κk²t} sin kx| ≤ |sin kx| in interval arithmetic at (κ, k, t, x) = (0, 1, 0, 1) (edge), (1, 2, 1, 1/3), (1/2, 3, 4, 2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '1', '0', '1', '_', '_',
                    lhs=abs(exp(-(0)*(1)^2*(0))*sin((1)*(1))),
                    rhs=abs(sin((1)*(1))), relation='<=', edge=True),
    oracle_instance('1', '2', '1', '1/3', '_', '_',
                    lhs=abs(exp(-(1)*(2)^2*(1))*sin((2)*(1/3))),
                    rhs=abs(sin((2)*(1/3))), relation='<=', edge=False),
    oracle_instance('1/2', '3', '4', '2', '_', '_',
                    lhs=abs(exp(-(1/2)*(3)^2*(4))*sin((3)*(2))),
                    rhs=abs(sin((3)*(2))), relation='<=', edge=False),
]
