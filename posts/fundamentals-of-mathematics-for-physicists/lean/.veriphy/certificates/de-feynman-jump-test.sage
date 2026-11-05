# name: de-feynman-jump-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FeynmanPrescription.feynman_jump file "de-feynman-jump-test.sage"
# description: the jump of dG/dt at t = 0 between the two branches ±sin(ωt)/(2ω), by Sage's derivatives, at ω = 1 (edge), 2, -1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '_',
                    lhs=(diff(sin((1)*s_)/(2*(1)), s_) - diff(-sin((1)*s_)/(2*(1)), s_)).subs(s_=0).simplify_full(),
                    rhs=1, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=(diff(sin((2)*s_)/(2*(2)), s_) - diff(-sin((2)*s_)/(2*(2)), s_)).subs(s_=0).simplify_full(),
                    rhs=1, relation='==', edge=False),
    oracle_instance('-1/3', '_',
                    lhs=(diff(sin((-1/3)*s_)/(2*(-1/3)), s_) - diff(-sin((-1/3)*s_)/(2*(-1/3)), s_)).subs(s_=0).simplify_full(),
                    rhs=1, relation='==', edge=False),
]
