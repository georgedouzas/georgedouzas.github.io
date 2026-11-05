# name: de-kernel-le-max-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.kernel_le_max file "de-kernel-le-max-test.sage"
# description: K(t, x) ≤ K(t, 0) in interval arithmetic at (κ, t, x) = (1, 1, 0) (edge), (2, 1/2, 3), (1/3, 2, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '0', '_', '_',
                    lhs=exp(-(0)^2/(4*(1)*(1)))/sqrt(4*pi*(1)*(1)),
                    rhs=1/sqrt(4*pi*(1)*(1)), relation='<=', edge=True),
    oracle_instance('2', '1/2', '3', '_', '_',
                    lhs=exp(-(3)^2/(4*(2)*(1/2)))/sqrt(4*pi*(2)*(1/2)),
                    rhs=1/sqrt(4*pi*(2)*(1/2)), relation='<=', edge=False),
    oracle_instance('1/3', '2', '-1', '_', '_',
                    lhs=exp(-(-1)^2/(4*(1/3)*(2)))/sqrt(4*pi*(1/3)*(2)),
                    rhs=1/sqrt(4*pi*(1/3)*(2)), relation='<=', edge=False),
]
