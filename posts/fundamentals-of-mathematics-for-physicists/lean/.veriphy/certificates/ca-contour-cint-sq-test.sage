# name: ca-contour-cint-sq-test
# technique: test fallback; oracle instances: Sage integrates z^2 along the path exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_sq
# description: instances of cint_sq (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(KaluzaKlein.ComplexAnalysis.Contours.seg 0 (1 + Complex.I))', t * (1 + I), '0', 0, '1', 1, False),
         ('KaluzaKlein.ComplexAnalysis.Contours.parabola', t + I * t ** 2, '0', 0, '1', 1, False),
         ('(fun s : ℝ => Complex.exp (s * Complex.I))', exp(I * t), '0', 0, 'Real.pi', pi, False),
         ('(fun s : ℝ => (s : ℂ))', t, '1', 1, '1', 1, True)]
instances = [oracle_instance(gl, al, bl, '_', lhs=ex(pint(z ** 2, g, a, b)), rhs=ex((g.subs(t=b) ** 3 - g.subs(t=a) ** 3) / 3), edge=e)
             for gl, g, al, a, bl, b, e in cases]
