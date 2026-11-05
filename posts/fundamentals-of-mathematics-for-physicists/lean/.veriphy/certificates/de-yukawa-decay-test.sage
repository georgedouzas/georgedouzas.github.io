# name: de-yukawa-decay-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensComputations.yukawa_decay file "de-yukawa-decay-test.sage"
# description: B ≠ 0 makes Ae^{-mr} + Be^{mr} unbounded: Sage's limit at ∞ is ±∞ for (A, B, m) = (1, 1, 1) (edge), (2, -3, 1/2), (0, 1/5, 4)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('1', '1', '1', '_', '_',
                    lhs=B(abs(limit((1)*exp(-(1)*x) + (1)*exp((1)*x), x=oo)) == oo),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '-3', '1/2', '_', '_',
                    lhs=B(abs(limit((2)*exp(-(1/2)*x) + (-3)*exp((1/2)*x), x=oo)) == oo),
                    rhs=1, relation='==', edge=False),
    oracle_instance('0', '1/5', '4', '_', '_',
                    lhs=B(abs(limit((0)*exp(-(4)*x) + (1/5)*exp((4)*x), x=oo)) == oo),
                    rhs=1, relation='==', edge=False),
]
