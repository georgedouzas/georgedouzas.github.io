# name: ca-contour-length-reverse-test
# technique: test fallback; oracle instances: Sage integrates |gamma'| for the path and its reversal exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Contours.length_reverse
# description: instances of length_reverse (Contour integration, part I chapter 4), edge case first or marked
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
def L(g, a, b):
    return integrate(speed(g), t, a, b)
cases = [('(fun s : ℝ => (s : ℂ) + Complex.I * (s : ℂ) ^ 2)', t + I * t ** 2, '0', '1', 0, 1, False),
         ('(fun s : ℝ => Complex.exp (s * Complex.I))', exp(I * t), '0', '(2 * Real.pi)', 0, 2 * pi, False),
         ('(fun s : ℝ => (s : ℂ) ^ 3)', t ** 3, '(-1)', '2', -1, 2, False),
         ('(fun s : ℝ => (s : ℂ))', t, '0', '0', 0, 0, True)]
instances = [oracle_instance(gl, al, bl, lhs=ex(L(g.subs(t=a + b - t), a, b)), rhs=ex(L(g, a, b)), edge=e)
             for gl, g, al, bl, a, b, e in cases]
