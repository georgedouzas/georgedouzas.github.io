# name: de-generating-equation-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LegendrePolynomials.generating_equation file "de-generating-equation-test.sage"
# description: (1 - 2xt + t²) ∂_t (1 - 2xt + t²)^{-1/2} against (x - t)(1 - 2xt + t²)^{-1/2} by Sage, exactly, at (x, t) = (0, 0) (edge), (1/2, 1/3), (-1, 2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0', '0', '_',
                    lhs=((1-2*(0)*s_+s_^2)*diff((1-2*(0)*s_+s_^2)^(-1/2), s_) - ((0)-s_)*(1-2*(0)*s_+s_^2)^(-1/2)).subs(s_=0).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1/2', '1/3', '_',
                    lhs=((1-2*(1/2)*s_+s_^2)*diff((1-2*(1/2)*s_+s_^2)^(-1/2), s_) - ((1/2)-s_)*(1-2*(1/2)*s_+s_^2)^(-1/2)).subs(s_=1/3).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1', '2', '_',
                    lhs=((1-2*(-1)*s_+s_^2)*diff((1-2*(-1)*s_+s_^2)^(-1/2), s_) - ((-1)-s_)*(1-2*(-1)*s_+s_^2)^(-1/2)).subs(s_=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
