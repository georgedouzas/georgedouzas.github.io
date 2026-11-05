# name: ca-contour-card-roots-test
# technique: test fallback; oracle instances: Sage counts the roots with multiplicity over the algebraic numbers
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.card_roots
# description: instances of card_roots (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(Polynomial.X ^ 2 + 1 : Polynomial ℂ)', x ** 2 + 1, False), ('((Polynomial.X - 1) ^ 2 * (Polynomial.X + 2) : Polynomial ℂ)', (x - 1) ** 2 * (x + 2), False),
         ('(3 : Polynomial ℂ)', x.parent()(3), True), ('(Polynomial.X ^ 5 - 3 * Polynomial.X + 2 : Polynomial ℂ)', x ** 5 - 3 * x + 2, False)]
instances = [oracle_instance(pl, lhs=QQ(sum(m for _, m in p.roots())), rhs=QQ(p.degree()), edge=e) for pl, p, e in cases]
