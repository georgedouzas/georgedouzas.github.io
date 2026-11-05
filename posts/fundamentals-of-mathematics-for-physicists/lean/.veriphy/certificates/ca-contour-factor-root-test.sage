# name: ca-contour-factor-root-test
# technique: test fallback; oracle instances: Sage divides by z - root over the algebraic numbers
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.factor_root
# description: instances of factor_root (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(Polynomial.X ^ 2 + 1 : Polynomial ℂ)', x ** 2 + 1, 'Complex.I', QQbar(I), False),
         ('(Polynomial.X ^ 3 - 1 : Polynomial ℂ)', x ** 3 - 1, '1', QQbar(1), False),
         ('(Polynomial.X : Polynomial ℂ)', x, '0', QQbar(0), True)]
instances = []
for pl, p, zl, z0, e in cases:
    q, rem = p.quo_rem(x - z0)
    instances.append(oracle_instance(pl, zl, '(by simp)', lhs=vector(QQbar, [rem, QQbar(q.degree() if q != 0 else 0)]), rhs=vector(QQbar, [0, p.degree() - 1]), edge=e))
