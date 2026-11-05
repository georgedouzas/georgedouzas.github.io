# name: de-saddle-axis-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.Stability.saddle_stable_axis file "de-saddle-axis-test.sage"
# description: Sage's limit of ce^{-t} at ∞ is 0 for c = 0 (edge), 3, -2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
instances = [
    oracle_instance('0',
                    lhs=limit((0)*exp(-x), x=oo),
                    rhs=0, relation='==', edge=True),
    oracle_instance('3',
                    lhs=limit((3)*exp(-x), x=oo),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-2',
                    lhs=limit((-2)*exp(-x), x=oo),
                    rhs=0, relation='==', edge=False),
]
