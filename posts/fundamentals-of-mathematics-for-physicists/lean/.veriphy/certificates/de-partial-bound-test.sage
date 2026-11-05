# name: de-partial-bound-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.partial_example_bound file "de-partial-bound-test.sage"
# description: |-x sin y + 3y²| ≤ |x| + 3y² in interval arithmetic at (0, 0) (edge), (2, 1), (-1, -3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0', '0',
                    lhs=RIF(0),
                    rhs=RIF(0), relation='<=', edge=True),
    oracle_instance('2', '1',
                    lhs=RIF(abs(-2*sin(1)+3)),
                    rhs=RIF(5), relation='<=', edge=False),
    oracle_instance('-1', '-3',
                    lhs=RIF(abs(sin(-3)+27)),
                    rhs=RIF(28), relation='<=', edge=False),
]
