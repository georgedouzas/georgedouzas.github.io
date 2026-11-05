# name: de-burgers-sum-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PartialDifferentialEquations.burgers_sum file "de-burgers-sum-test.sage"
# description: u_t + u u_x for u = 1 + x/(1 + t) by Sage's partial derivatives, exactly, at (t, x) = (0, 1) (edge), (1, 2), (-1/2, -3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0', '1', '_',
                    lhs=(diff(1+xs/(1+tt), tt) + (1+xs/(1+tt))*diff(1+xs/(1+tt), xs)).subs(tt=0, xs=1).simplify_full(),
                    rhs=1/(1+(0)), relation='==', edge=True),
    oracle_instance('1', '2', '_',
                    lhs=(diff(1+xs/(1+tt), tt) + (1+xs/(1+tt))*diff(1+xs/(1+tt), xs)).subs(tt=1, xs=2).simplify_full(),
                    rhs=1/(1+(1)), relation='==', edge=False),
    oracle_instance('-1/2', '-3', '_',
                    lhs=(diff(1+xs/(1+tt), tt) + (1+xs/(1+tt))*diff(1+xs/(1+tt), xs)).subs(tt=-1/2, xs=-3).simplify_full(),
                    rhs=1/(1+(-1/2)), relation='==', edge=False),
]
