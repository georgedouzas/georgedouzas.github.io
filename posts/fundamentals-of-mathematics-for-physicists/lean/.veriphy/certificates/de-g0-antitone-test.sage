# name: de-g0-antitone-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.g0_antitone file "de-g0-antitone-test.sage"
# description: gᵢ₊₁ ≤ gᵢ for 0 ≤ x ≤ 4, exactly for i < 20 at x = 0 (edge), 2, 4
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def g(i, x): return (x/2)^(2*i+2)/factorial(i+1)^2
instances = [
    oracle_instance('0', '_', '_',
                    lhs=B(all(g(i+1, QQ(0)) <= g(i, QQ(0)) for i in range(20))),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '_', '_',
                    lhs=B(all(g(i+1, QQ(2)) <= g(i, QQ(2)) for i in range(20))),
                    rhs=1, relation='==', edge=False),
    oracle_instance('4', '_', '_',
                    lhs=B(all(g(i+1, QQ(4)) <= g(i, QQ(4)) for i in range(20))),
                    rhs=1, relation='==', edge=False),
]
