# name: de-drum-mode-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.BesselFunctions.drum_mode file "de-drum-mode-test.sage"
# description: u = F(jx) for F = J₀ satisfies x²u'' + xu' + j²x²u = 0: Sage's bessel_J at (j, x) = (1, 1) (edge), (2, 1/2), (5, 3/10), residual below 10⁻²⁰
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
y_ = var('y_')
instances = [
    oracle_instance('1', '1', '_', '_', '_', '_', '_', '_',
                    lhs=RIF(abs(((1)^2*diff(bessel_J(0, (1)*y_), y_, 2) + (1)*diff(bessel_J(0, (1)*y_), y_) + (1)^2*(1)^2*bessel_J(0, (1)*y_)).subs(y_=1).n(prec=120))),
                    rhs=RIF(10^-20), relation='<', edge=True),
    oracle_instance('2', '1/2', '_', '_', '_', '_', '_', '_',
                    lhs=RIF(abs(((1/2)^2*diff(bessel_J(0, (2)*y_), y_, 2) + (1/2)*diff(bessel_J(0, (2)*y_), y_) + (2)^2*(1/2)^2*bessel_J(0, (2)*y_)).subs(y_=1/2).n(prec=120))),
                    rhs=RIF(10^-20), relation='<', edge=False),
    oracle_instance('5', '3/10', '_', '_', '_', '_', '_', '_',
                    lhs=RIF(abs(((3/10)^2*diff(bessel_J(0, (5)*y_), y_, 2) + (3/10)*diff(bessel_J(0, (5)*y_), y_) + (5)^2*(3/10)^2*bessel_J(0, (5)*y_)).subs(y_=3/10).n(prec=120))),
                    rhs=RIF(10^-20), relation='<', edge=False),
]
