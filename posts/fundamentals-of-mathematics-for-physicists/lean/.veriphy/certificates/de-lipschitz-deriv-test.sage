# name: de-lipschitz-deriv-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.lipschitz_of_deriv_bound file "de-lipschitz-deriv-test.sage"
# description: |g(a) - g(b)| ≤ L|a - b| for g = sin (L = 1), g = 3x (L = 3), g = 0 (edge), in interval arithmetic
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('Real.sin', 'Real.cos', '1', '_', '_', '2', '-1',
                    lhs=RIF(abs(sin(2)-sin(-1))),
                    rhs=RIF(3), relation='<=', edge=False),
    oracle_instance('fun x => 3 * x', 'fun _ => 3', '3', '_', '_', '1/2', '5',
                    lhs=abs(3/2-15),
                    rhs=3*abs(1/2-5), relation='<=', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 0', '0', '_', '_', '0', '0',
                    lhs=0,
                    rhs=0, relation='<=', edge=True),
]
