# name: de-kernel-solves-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.HeatEquation.kernel_solves file "de-kernel-solves-test.sage"
# description: K_t - κ K_xx for the heat kernel by Sage's partial derivatives, in interval arithmetic, at (κ, t, x) = (1, 1, 0) (edge), (2, 1/2, 1), (1/3, 3, -2)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '1', '0', '_', '_', '_', '_',
                    lhs=RIF(abs((diff(exp(-xs^2/(4*(1)*tt))/sqrt(4*pi*(1)*tt), tt) - (1)*diff(exp(-xs^2/(4*(1)*tt))/sqrt(4*pi*(1)*tt), xs, 2)).subs(tt=1, xs=0))),
                    rhs=RIF(10^-15), relation='<', edge=True),
    oracle_instance('2', '1/2', '1', '_', '_', '_', '_',
                    lhs=RIF(abs((diff(exp(-xs^2/(4*(2)*tt))/sqrt(4*pi*(2)*tt), tt) - (2)*diff(exp(-xs^2/(4*(2)*tt))/sqrt(4*pi*(2)*tt), xs, 2)).subs(tt=1/2, xs=1))),
                    rhs=RIF(10^-15), relation='<', edge=False),
    oracle_instance('1/3', '3', '-2', '_', '_', '_', '_',
                    lhs=RIF(abs((diff(exp(-xs^2/(4*(1/3)*tt))/sqrt(4*pi*(1/3)*tt), tt) - (1/3)*diff(exp(-xs^2/(4*(1/3)*tt))/sqrt(4*pi*(1/3)*tt), xs, 2)).subs(tt=3, xs=-2))),
                    rhs=RIF(10^-15), relation='<', edge=False),
]
