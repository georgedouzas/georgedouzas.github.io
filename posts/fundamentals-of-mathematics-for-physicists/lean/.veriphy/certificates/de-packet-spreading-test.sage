# name: de-packet-spreading-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.PDEComputations.packet_spreading file "de-packet-spreading-test.sage"
# description: ω''(k) for ω = √(k² + m²) against m²/ω³ by Sage, in interval arithmetic, at (k, m) = (0, 1) (edge), (1, 2), (-3/2, 1/2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('0', '1', '_',
                    lhs=RIF(abs((diff(sqrt(ks^2+(1)^2), ks, 2) - (1)^2/sqrt(ks^2+(1)^2)^3).subs(ks=0))),
                    rhs=RIF(10^-15), relation='<', edge=True),
    oracle_instance('1', '2', '_',
                    lhs=RIF(abs((diff(sqrt(ks^2+(2)^2), ks, 2) - (2)^2/sqrt(ks^2+(2)^2)^3).subs(ks=1))),
                    rhs=RIF(10^-15), relation='<', edge=False),
    oracle_instance('-3/2', '1/2', '_',
                    lhs=RIF(abs((diff(sqrt(ks^2+(1/2)^2), ks, 2) - (1/2)^2/sqrt(ks^2+(1/2)^2)^3).subs(ks=-3/2))),
                    rhs=RIF(10^-15), relation='<', edge=False),
]
