# name: de-gronwall-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.gronwall_decay file "de-gronwall-test.sage"
# description: f(t) ≤ e^{-2ct}f(0) for f' ≤ -2cf: f = e^{-3t} (c = 1), f = 0 (edge), f = e^{-2t}/(1 + t) (c = 1), at t = 1, 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('fun t => Real.exp (-3 * t)', '_', '1', '_', '_', '1', '_',
                    lhs=RIF(exp(-3)),
                    rhs=RIF(exp(-2)), relation='<=', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 0', '1', '_', '_', '1', '_',
                    lhs=0,
                    rhs=0, relation='<=', edge=True),
    oracle_instance('fun t => Real.exp (-2 * t) / (1 + t ^ 2)', '_', '1', '_', '_', '2', '_',
                    lhs=RIF(exp(-4)/5),
                    rhs=RIF(exp(-4)), relation='<=', edge=False),
]
