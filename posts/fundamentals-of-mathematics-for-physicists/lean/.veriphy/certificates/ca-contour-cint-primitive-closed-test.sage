# name: ca-contour-cint-primitive-closed-test
# technique: test fallback; oracle instances: Sage integrates f around closed paths exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_primitive_closed
# description: instances of cint_primitive_closed (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun w : ℂ => w ^ 2)', z ** 2, '(fun w : ℂ => w ^ 3 / 3)', '(fun s : ℝ => Complex.exp (s * Complex.I))', exp(I * t), '0', 0, '(2 * Real.pi)', 2 * pi, False),
         ('(fun w : ℂ => w ^ 3)', z ** 3, '(fun w : ℂ => w ^ 4 / 4)', '(fun s : ℝ => 2 * Real.cos s + Complex.I * Real.sin s)', 2 * cos(t) + I * sin(t), '0', 0, '(2 * Real.pi)', 2 * pi, False),
         ('(fun w : ℂ => w)', z, '(fun w : ℂ => w ^ 2 / 2)', '(fun _ : ℝ => (1 : ℂ))', SR(1), '0', 0, '1', 1, True)]
instances = [oracle_instance(fl, Fl, gl, al, bl, '_', '_', '_', '_', lhs=ex(pint(f, g, a, b)), rhs=SR(0), edge=e)
             for fl, f, Fl, gl, g, al, a, bl, b, e in cases]
