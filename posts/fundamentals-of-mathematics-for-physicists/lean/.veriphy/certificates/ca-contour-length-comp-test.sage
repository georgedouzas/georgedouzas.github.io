# name: ca-contour-length-comp-test
# technique: test fallback; oracle instances: Sage computes both lengths in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.Contours.length_comp
# description: instances of length_comp (Contour integration, part I chapter 4), edge case first or marked
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
def Lb(sp, a, b):
    # length from the (analytic) speed function sp on balls
    return bint(sp, a, b)
# gamma(t) = t + i t^2 has speed sqrt(1 + 4t^2); phi = s^3 + s (phi' = 3 s^2 + 1), phi = 2 s, phi = s (edge)
dpar = lambda x: (1 + 4 * x ** 2).sqrt(analytic=True)
P = '(fun s : ℝ => (s : ℂ) + Complex.I * (s : ℂ) ^ 2)'
cases = [('(fun s : ℝ => s ^ 3 + s)', '(fun s : ℝ => 3 * s ^ 2 + 1)', lambda s: s ** 3 + s, lambda s: 3 * s ** 2 + 1, 0, 1, False),
         ('(fun s : ℝ => 2 * s)', '(fun _ : ℝ => 2)', lambda s: 2 * s, lambda s: 2, 0, 1, False),
         ('(fun s : ℝ => s)', '(fun _ : ℝ => 1)', lambda s: s, lambda s: 1, -1, 1, True)]
instances = []
for fl, dl, phi, dphi, c, d, e in cases:
    lhs = Lb(lambda x: dpar(phi(x)) * dphi(x), c, d)
    rhs = Lb(dpar, phi(QQ(c)), phi(QQ(d)))
    instances.append(oracle_instance(P, fl, dl, str(c), str(d), '_', '_', '_', '_', '_', lhs=gap((lhs, rhs)), rhs=RI(0), edge=e))
