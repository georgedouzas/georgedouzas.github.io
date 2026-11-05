# name: ca-contour-poly-lower-bound-test
# technique: test fallback; oracle instances: Sage computes the radius and compares both sides at points beyond it in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.poly_lower_bound
# description: instances of poly_lower_bound (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(Polynomial.X ^ 2 + 1 : Polynomial ℂ)', x ** 2 + 1, '(3 : ℂ)', CB(3), False),
         ('(Polynomial.X ^ 3 - 5 * Polynomial.X + 2 : Polynomial ℂ)', x ** 3 - 5 * x + 2, '(20 * Complex.I)', CB(0, 20), False),
         ('(2 * Polynomial.X : Polynomial ℂ)', 2 * x, '(1 : ℂ)', CB(1), True)]
instances = []
for pl, p, zl, z0, e in cases:
    n = p.degree()
    a = p.leading_coefficient()
    R = max(1, 2 * sum(abs(p[k]) for k in range(n)) / abs(a))
    assert RI(abs(z0)) >= RI(R), 'the instance point must lie beyond the radius'
    val = sum(CB(p[k]) * z0 ** k for k in range(n + 1))
    instances.append(oracle_instance(pl, '(by compute_degree!)', zl, '_', lhs=RI(abs(a)) * RI(abs(z0)) ** n / 2, rhs=RI(abs(val)), relation='<=', edge=e))
