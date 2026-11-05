# name: ca-contour-cint-re-im-test
# technique: test fallback; oracle instances: Sage integrates the integrand and its real and imaginary parts exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_re_im
# description: instances of cint_re_im (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun w : ℂ => w ^ 2)', z ** 2, '(fun s : ℝ => (s : ℂ) + Complex.I * (s : ℂ) ^ 2)', t + I * t ** 2, '0', 0, '1', 1, False),
         ('(starRingEnd ℂ)', conjugate(z), '(fun s : ℝ => (s : ℂ) * (1 + Complex.I))', t * (1 + I), '0', 0, '1', 1, False),
         ('(fun w : ℂ => w)', z, '(fun s : ℝ => (s : ℂ))', t, '0', 0, '0', 0, True)]
instances = []
for fl, f, gl, g, al, a, bl, b, e in cases:
    h = (f.subs(z=g) * diff(g, t)).rectform()
    lhs = integrate(h, t, a, b)
    rhs = integrate(h.real_part(), t, a, b) + I * integrate(h.imag_part(), t, a, b)
    instances.append(oracle_instance(fl, gl, al, bl, '_', lhs=ex(lhs), rhs=ex(rhs), edge=e))
