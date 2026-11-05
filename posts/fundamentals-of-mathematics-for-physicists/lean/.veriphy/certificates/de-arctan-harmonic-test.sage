# name: de-arctan-harmonic-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.arctan_harmonic file "de-arctan-harmonic-test.sage"
# description: ∇² arctan(y/x) = 0 by Sage at (x, y) = (1, 0) (edge), (2, 3), (-1, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
y_ = var('y_')
instances = [
    oracle_instance('1', '0', '_',
                    lhs=(diff(arctan(y_/x), x, 2) + diff(arctan(y_/x), y_, 2)).simplify_full().subs(x=1, y_=0),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '3', '_',
                    lhs=(diff(arctan(y_/x), x, 2) + diff(arctan(y_/x), y_, 2)).simplify_full().subs(x=2, y_=3),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1', '1/2', '_',
                    lhs=(diff(arctan(y_/x), x, 2) + diff(arctan(y_/x), y_, 2)).simplify_full().subs(x=-1, y_=1/2),
                    rhs=0, relation='==', edge=False),
]
