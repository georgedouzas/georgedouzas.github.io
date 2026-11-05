# name: de-pendulum-lipschitz-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.InitialValueProblems.pendulum_lipschitz file "de-pendulum-lipschitz-test.sage"
# description: |sin a - sin b| ≤ |a - b| in interval arithmetic at (0, 0) (edge), (1, 2), (-3, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=RIF(abs(sin(0)-sin(0))),
                    rhs=RIF(0), relation='<=', edge=True),
    oracle_instance('1', '2',
                    lhs=RIF(abs(sin(1)-sin(2))),
                    rhs=RIF(1), relation='<=', edge=False),
    oracle_instance('-3', '1/2',
                    lhs=RIF(abs(sin(-3)-sin(1/2))),
                    rhs=RIF(7/2), relation='<=', edge=False),
]
