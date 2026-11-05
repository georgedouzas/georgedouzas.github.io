# name: de-sink-decay-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.sink_decay file "de-sink-decay-test.sage"
# description: |e^{-t}(cos t + sin t)| ≤ 2e^{-t} in interval arithmetic at t = 0 (edge), 1, -2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=RIF(abs(exp(-(0))*(cos(0)+sin(0)))),
                    rhs=RIF(2*exp(-(0))), relation='<=', edge=True),
    oracle_instance('1',
                    lhs=RIF(abs(exp(-(1))*(cos(1)+sin(1)))),
                    rhs=RIF(2*exp(-(1))), relation='<=', edge=False),
    oracle_instance('-2',
                    lhs=RIF(abs(exp(-(-2))*(cos(-2)+sin(-2)))),
                    rhs=RIF(2*exp(-(-2))), relation='<=', edge=False),
]
