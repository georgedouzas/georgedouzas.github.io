# name: de-screening-charge-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.GreensComputations.screening_charge file "de-screening-charge-test.sage"
# description: ∫₀ᴿ m² r e^{-mr} dr by Sage's integrate against 1 - e^{-mR}(1 + mR), exactly, at (m, R) = (1, 0) (edge), (2, 1), (1/2, 3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '0', '_',
                    lhs=(integrate((1)^2*rs*exp(-(1)*rs), rs, 0, 0) - (1 - exp(-(1)*(0))*(1 + (1)*(0)))).subs(rs=rs).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '1', '_',
                    lhs=(integrate((2)^2*rs*exp(-(2)*rs), rs, 0, 1) - (1 - exp(-(2)*(1))*(1 + (2)*(1)))).subs(rs=rs).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('1/2', '3', '_',
                    lhs=(integrate((1/2)^2*rs*exp(-(1/2)*rs), rs, 0, 3) - (1 - exp(-(1/2)*(3))*(1 + (1/2)*(3)))).subs(rs=rs).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
