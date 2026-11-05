# name: ca-contour-cauchy-rect-test
# technique: test fallback; oracle instances: Sage encloses the four side integrals in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.cauchy_rect
# description: instances of cauchy_rect (Contour integration, part I chapter 4), edge case first or marked
SEED = 0
t = var('t', domain='real')
z = var('z')
CB = ComplexBallField(128)
RI = RealIntervalField(128)
J = CB(0, 1)
def ex(e):
    return SR(e).rectform().simplify_full()
def pint(f, g, a, b):
    # the parametrised contour integral of f (an expression in z) along g(t), t in [a, b], exactly
    return integrate(f.subs(z=g) * diff(g, t), t, a, b)
def speed(g):
    d = diff(g, t)
    return sqrt(d.real_part() ** 2 + d.imag_part() ** 2).simplify_full()
def bint(F, a, b):
    # a rigorous ball enclosure of the integral of F (a function on balls) over [a, b]
    return CB.integral(lambda x, _: F(x), CB(a), CB(b))
def circ(F, c, R):
    # a rigorous ball enclosure of the circle integral of F over |z - c| = R
    return bint(lambda x: F(CB(c) + CB(R) * (J * x).exp()) * CB(R) * J * (J * x).exp(), 0, 2 * CB.pi())
def gap(*pairs):
    # an interval enclosing the sum of |x - y| over the pairs: it contains 0 when every pair agrees
    return sum(RI(abs(CB(x) - CB(y))) for x, y in pairs)
def rect(F, z0, w0):
    x0, y0, x1, y1 = CB(z0).real(), CB(z0).imag(), CB(w0).real(), CB(w0).imag()
    a = bint(lambda x: F(x + y0 * J), x0, x1)
    b = bint(lambda x: F(x + y1 * J), x0, x1)
    c = bint(lambda y: F(x1 + y * J), y0, y1)
    d = bint(lambda y: F(x0 + y * J), y0, y1)
    return a - b + J * c - J * d
cases = [('Complex.exp', lambda w: w.exp(), '0', 0, '(1 + Complex.I)', 1 + I, False),
         ('(fun w : ℂ => w ^ 3)', lambda w: w ** 3, '(-1 - 2 * Complex.I)', -1 - 2 * I, '(2 + Complex.I)', 2 + I, False),
         ('Complex.sin', lambda w: w.sin(), '(1 + Complex.I)', 1 + I, '(1 + Complex.I)', 1 + I, True)]
instances = [oracle_instance(fl, zl, wl, '_', lhs=gap((rect(F, z0, w0), 0)), rhs=RI(0), edge=e) for fl, F, zl, z0, wl, w0, e in cases]
