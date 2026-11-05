# name: de-dissipation-nonpos-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.dissipation_nonpos file "de-dissipation-nonpos-test.sage"
# description: -γq̇² ≤ 0 at (γ, q̇) = (0, 0) (edge), (1/2, 3), (2, -1)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0', '_',
                    lhs=-(0)*(0)^2,
                    rhs=0, relation='<=', edge=True),
    oracle_instance('1/2', '3', '_',
                    lhs=-(1/2)*(3)^2,
                    rhs=0, relation='<=', edge=False),
    oracle_instance('2', '-1', '_',
                    lhs=-(2)*(-1)^2,
                    rhs=0, relation='<=', edge=False),
]
