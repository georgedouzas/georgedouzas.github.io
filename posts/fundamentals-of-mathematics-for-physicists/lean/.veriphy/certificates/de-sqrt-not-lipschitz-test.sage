# name: de-sqrt-not-lipschitz-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.InitialValueProblems.sqrt_not_lipschitz file "de-sqrt-not-lipschitz-test.sage"
# description: the witness y = 1/(|L| + 1)² gives L y < 2√y, at L = 0 (edge), 3, -5
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=RIF(0*(1/1)^2),
                    rhs=RIF(2*sqrt(1)), relation='<', edge=True),
    oracle_instance('3',
                    lhs=RIF(3/16),
                    rhs=RIF(2*sqrt(1/16)), relation='<', edge=False),
    oracle_instance('-5',
                    lhs=RIF(-5/36),
                    rhs=RIF(2*sqrt(1/36)), relation='<', edge=False),
]
