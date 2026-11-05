# name: de-blowup-general-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.ExistenceUniqueness.blowup_general_solves file "de-blowup-general-solves-test.sage"
# description: (y₀/(1 - y₀x))' = (y₀/(1 - y₀x))² by Sage at (y₀, x) = (0, 1) (edge), (2, 1/3), (-1, 2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('0', '1', '_',
                    lhs=(diff((0)/(1-(0)*s_), s_) - (((0)/(1-(0)*s_))^2)).subs(s_=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '1/3', '_',
                    lhs=(diff((2)/(1-(2)*s_), s_) - (((2)/(1-(2)*s_))^2)).subs(s_=1/3).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1', '2', '_',
                    lhs=(diff((-1)/(1-(-1)*s_), s_) - (((-1)/(1-(-1)*s_))^2)).subs(s_=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
