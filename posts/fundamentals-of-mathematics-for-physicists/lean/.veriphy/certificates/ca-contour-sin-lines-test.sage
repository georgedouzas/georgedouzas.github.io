# name: ca-contour-sin-lines-test
# technique: test fallback; oracle instances: Sage compares |sin(x + iy)|^2 with sinh^2 y and 1 + sinh^2 y in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.sin_lines
# description: instances of sin_lines (Contour integration, part I chapter 4), edge case first or marked
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
instances = []
for xl, x0, yl, y0, e in [('0', 0, '0', 0, True), ('1', 1, '2', 2, False), ('(-5 / 2)', -QQ(5) / 2, '(1 / 3)', QQ(1) / 3, False), ('3', 3, '(-4)', -4, False)]:
    s2 = RI(abs(CB(x0, y0).sin())) ** 2
    sh2 = RI(y0).sinh() ** 2
    instances.append(oracle_instance(xl, yl, lhs=(sh2 - s2).max(s2 - 1 - sh2), rhs=RI(0), relation='<=', edge=e))
