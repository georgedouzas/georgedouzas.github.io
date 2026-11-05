# name: de-yukawa-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.yukawa_solves file "de-yukawa-solves-test.sage"
# description: (-∇² + m²) of e^{-mr}/(4πr) by Sage's radial derivatives, exactly, at (m, r) = (0, 1) (edge), (1, 2), (3/2, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0', '1', '_',
                    lhs=(-(diff(exp(-ms*rs)/(4*pi*rs), rs, 2) + 2/rs*diff(exp(-ms*rs)/(4*pi*rs), rs)) + ms^2*exp(-ms*rs)/(4*pi*rs)).subs(ms=0, rs=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1', '2', '_',
                    lhs=(-(diff(exp(-ms*rs)/(4*pi*rs), rs, 2) + 2/rs*diff(exp(-ms*rs)/(4*pi*rs), rs)) + ms^2*exp(-ms*rs)/(4*pi*rs)).subs(ms=1, rs=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('3/2', '1/2', '_',
                    lhs=(-(diff(exp(-ms*rs)/(4*pi*rs), rs, 2) + 2/rs*diff(exp(-ms*rs)/(4*pi*rs), rs)) + ms^2*exp(-ms*rs)/(4*pi*rs)).subs(ms=3/2, rs=1/2).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
