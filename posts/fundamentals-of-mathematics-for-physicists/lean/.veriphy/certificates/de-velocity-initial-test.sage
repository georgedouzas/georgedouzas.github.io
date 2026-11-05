# name: de-velocity-initial-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.velocity_initial file "de-velocity-initial-test.sage"
# description: ∂_t [cos x sin(ct)/c] at t = 0 by Sage, exactly, at (c, x) = (1, 0) (edge), (2, 1), (-1/2, 3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '0', '_',
                    lhs=(diff(cos(0)*sin((1)*s_)/(1), s_)).subs(s_=0).simplify_full(),
                    rhs=cos(0), relation='==', edge=True),
    oracle_instance('2', '1', '_',
                    lhs=(diff(cos(1)*sin((2)*s_)/(2), s_)).subs(s_=0).simplify_full(),
                    rhs=cos(1), relation='==', edge=False),
    oracle_instance('-1/2', '3', '_',
                    lhs=(diff(cos(3)*sin((-1/2)*s_)/(-1/2), s_)).subs(s_=0).simplify_full(),
                    rhs=cos(3), relation='==', edge=False),
]
