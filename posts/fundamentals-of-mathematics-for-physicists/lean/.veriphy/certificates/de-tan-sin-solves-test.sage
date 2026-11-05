# name: de-tan-sin-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.SeparableEquations.tan_sin_solves file "de-tan-sin-solves-test.sage"
# description: y = tan(sin x) has y' = (1 + y²) cos x by Sage at x = 0 (edge), 1, -2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
s_ = var('s_')
instances = [
    oracle_instance('fun s => Real.sin (Real.sin s) / Real.cos (Real.sin s)', '_', '0',
                    lhs=((diff(sin(sin(s_))/cos(sin(s_)), s_) - ((1 + (sin(sin(s_))/cos(sin(s_)))^2)*cos(s_))).subs(s_=0).simplify_full()).simplify_trig(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('fun s => Real.sin (Real.sin s) / Real.cos (Real.sin s)', '_', '1',
                    lhs=((diff(sin(sin(s_))/cos(sin(s_)), s_) - ((1 + (sin(sin(s_))/cos(sin(s_)))^2)*cos(s_))).subs(s_=1).simplify_full()).simplify_trig(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('fun s => Real.sin (Real.sin s) / Real.cos (Real.sin s)', '_', '-2',
                    lhs=((diff(sin(sin(s_))/cos(sin(s_)), s_) - ((1 + (sin(sin(s_))/cos(sin(s_)))^2)*cos(s_))).subs(s_=-2).simplify_full()).simplify_trig(),
                    rhs=0, relation='==', edge=False),
]
