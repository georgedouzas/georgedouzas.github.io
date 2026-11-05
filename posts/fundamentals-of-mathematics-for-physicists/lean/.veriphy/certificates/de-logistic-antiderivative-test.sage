# name: de-logistic-antiderivative-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.logistic_antiderivative file "de-logistic-antiderivative-test.sage"
# description: (log(y/(1 - y)))' = 1/(y(1 - y)) by Sage at y = 1/2 (edge, symmetric point), 1/10, 9/10
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('1/2', '_', '_',
                    lhs=(diff(log(s_/(1-s_)), s_) - (1/(s_*(1-s_)))).subs(s_=1/2).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1/10', '_', '_',
                    lhs=(diff(log(s_/(1-s_)), s_) - (1/(s_*(1-s_)))).subs(s_=1/10).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('9/10', '_', '_',
                    lhs=(diff(log(s_/(1-s_)), s_) - (1/(s_*(1-s_)))).subs(s_=9/10).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
