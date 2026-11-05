# name: ca-contour-entire-closed-integrals-test
# technique: test fallback; oracle instances: Sage encloses the three closed integrals in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.entire_closed_integrals
# description: instances of entire_closed_integrals (Contour integration, part I chapter 4), edge case first or marked
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
def ints(p, g, dg, a, b):
    return [bint(lambda x: F(g(x)) * dg(x), a, b) for F in (lambda w: w.exp(), lambda w: w.cos(), p)]
circle = (lambda x: (J * x).exp(), lambda x: J * (J * x).exp())
ellipse = (lambda x: 2 * x.cos() + J * x.sin(), lambda x: -2 * x.sin() + J * x.cos())
point = (lambda x: CB(1), lambda x: CB(0))
cases = [('(Polynomial.X ^ 5 - 3 * Polynomial.X + 2 : Polynomial ℂ)', lambda w: w ** 5 - 3 * w + 2, '(fun s : ℝ => Complex.exp (s * Complex.I))', circle, False),
         ('(Polynomial.X ^ 2 : Polynomial ℂ)', lambda w: w ** 2, '(fun s : ℝ => 2 * Real.cos s + Complex.I * Real.sin s)', ellipse, False),
         ('(1 : Polynomial ℂ)', lambda w: CB(1), '(fun _ : ℝ => (1 : ℂ))', point, True)]
instances = [oracle_instance(pl, gl, '0', '(2 * Real.pi)', '_', '_', lhs=gap(*[(v, 0) for v in ints(p, g[0], g[1], 0, 2 * CB.pi())]), rhs=RI(0), edge=e)
             for pl, p, gl, g, e in cases]
