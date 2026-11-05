# name: de-coulomb-harmonic-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.FundamentalSolutions.coulomb_harmonic file "de-coulomb-harmonic-test.sage"
# description: the radial Laplacian of -1/(4πr) by Sage's derivatives, exactly, at r = 1 (edge), 2, 1/3
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '_',
                    lhs=(diff(-1/(4*pi*rs), rs, 2) + 2/rs*diff(-1/(4*pi*rs), rs)).subs(rs=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '_',
                    lhs=(diff(-1/(4*pi*rs), rs, 2) + 2/rs*diff(-1/(4*pi*rs), rs)).subs(rs=2).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('1/3', '_',
                    lhs=(diff(-1/(4*pi*rs), rs, 2) + 2/rs*diff(-1/(4*pi*rs), rs)).subs(rs=1/3).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
