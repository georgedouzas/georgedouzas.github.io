# name: de-log-harmonic-pde-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.LaplaceEquation.log_harmonic file "de-log-harmonic-pde-test.sage"
# description: the Laplacian of log(x² + y²) by Sage's partial derivatives, exactly, at (x, y) = (1, 1) (edge), (2, -1), (-1/2, 3)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
xs, ys, zs, rs, s_, th, ph, xi, tt, kk, ms, ks = var('xs ys zs rs s_ th ph xi tt kk ms ks')
instances = [
    oracle_instance('1', '1', '_', '_',
                    lhs=(diff(log(xs^2+ys^2), xs, 2) + diff(log(xs^2+ys^2), ys, 2)).subs(xs=1, ys=1).simplify_full(),
                    rhs=0, relation='==', edge=True),
    oracle_instance('2', '-1', '_', '_',
                    lhs=(diff(log(xs^2+ys^2), xs, 2) + diff(log(xs^2+ys^2), ys, 2)).subs(xs=2, ys=-1).simplify_full(),
                    rhs=0, relation='==', edge=False),
    oracle_instance('-1/2', '3', '_', '_',
                    lhs=(diff(log(xs^2+ys^2), xs, 2) + diff(log(xs^2+ys^2), ys, 2)).subs(xs=-1/2, ys=3).simplify_full(),
                    rhs=0, relation='==', edge=False),
]
