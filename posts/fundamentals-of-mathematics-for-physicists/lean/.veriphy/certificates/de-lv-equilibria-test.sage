# name: de-lv-equilibria-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PhasePortraits.lv_equilibria file "de-lv-equilibria-test.sage"
# description: Sage's solve of the Lotka-Volterra equilibrium equations gives exactly (0, 0) and (1, 1), checked at (0, 0) (edge), (1, 1) and the full solution set
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
y_ = var('y_')
instances = [
    oracle_instance('0', '0', '_', '_',
                    lhs=B((0 == 0 and 0 == 0)),
                    rhs=1, relation='==', edge=True),
    oracle_instance('1', '1', '_', '_',
                    lhs=B((1 == 1 and 1 == 1)),
                    rhs=1, relation='==', edge=False),
    oracle_instance('_', '_', '_', '_',
                    lhs=len(solve([x - x*y_ == 0, -y_ + x*y_ == 0], x, y_)),
                    rhs=2, relation='==', edge=False),
]
