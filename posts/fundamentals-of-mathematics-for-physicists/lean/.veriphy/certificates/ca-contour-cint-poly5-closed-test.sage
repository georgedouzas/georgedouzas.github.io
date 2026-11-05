# name: ca-contour-cint-poly5-closed-test
# technique: test fallback; oracle instances: Sage integrates z^5 - 3z + 2 around closed paths exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.cint_poly5_closed
# description: instances of cint_poly5_closed (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun s : ℝ => Complex.exp (s * Complex.I))', exp(I * t), '0', 0, '(2 * Real.pi)', 2 * pi, False),
         ('(fun s : ℝ => 2 * Real.cos s + Complex.I * Real.sin s)', 2 * cos(t) + I * sin(t), '0', 0, '(2 * Real.pi)', 2 * pi, False),
         ('(fun _ : ℝ => (1 : ℂ))', SR(1), '0', 0, '1', 1, True)]
instances = [oracle_instance(gl, al, bl, '_', '_', lhs=ex(pint(z ** 5 - 3 * z + 2, g, a, b)), rhs=SR(0), edge=e) for gl, g, al, a, bl, b, e in cases]
