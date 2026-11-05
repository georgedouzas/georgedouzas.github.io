# name: de-green-sine-left-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensBoundary.green_sine_left file "de-green-sine-left-test.sage"
# description: ∫₀ˣ ξ(1-x) sin πξ dξ by Sage's integrate against the closed form, exactly, at x = 0 (edge), 1/2, 2
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0',
                    lhs=(integrate(xi*(1-(0))*sin(pi*xi), xi, 0, 0) - (1-(0))*(sin(pi*(0))/pi^2 - (0)*cos(pi*(0))/pi)).subs(xi=xi).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('1/2',
                    lhs=(integrate(xi*(1-(1/2))*sin(pi*xi), xi, 0, 1/2) - (1-(1/2))*(sin(pi*(1/2))/pi^2 - (1/2)*cos(pi*(1/2))/pi)).subs(xi=xi).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('2',
                    lhs=(integrate(xi*(1-(2))*sin(pi*xi), xi, 0, 2) - (1-(2))*(sin(pi*(2))/pi^2 - (2)*cos(pi*(2))/pi)).subs(xi=xi).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
