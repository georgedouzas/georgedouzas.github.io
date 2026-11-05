# name: ca-contour-poly-factors-test
# technique: test fallback; oracle instances: Sage factors the polynomial over the algebraic numbers
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.poly_factors
# description: instances of poly_factors (Contour integration, part I chapter 4), edge case first or marked
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
x = polygen(QQbar, 'x')
def factored(p):
    return p.leading_coefficient() * prod([(x - r) ** m for r, m in p.roots()], x.parent()(1))
cases = [('(Polynomial.X ^ 2 + 1 : Polynomial ℂ)', x ** 2 + 1, False), ('(2 * Polynomial.X ^ 3 - 2 : Polynomial ℂ)', 2 * x ** 3 - 2, False),
         ('(3 : Polynomial ℂ)', x.parent()(3), True), ('((Polynomial.X - 1) ^ 2 * (Polynomial.X + 2) : Polynomial ℂ)', (x - 1) ** 2 * (x + 2), False)]
instances = [oracle_instance(pl, lhs=vector(QQbar, (factored(p) - p).padded_list(4)), rhs=vector(QQbar, [0] * 4), edge=e) for pl, p, e in cases]
