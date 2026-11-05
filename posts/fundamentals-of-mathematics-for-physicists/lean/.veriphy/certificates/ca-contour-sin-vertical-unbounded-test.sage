# name: ca-contour-sin-vertical-unbounded-test
# technique: test fallback; oracle instances: Sage evaluates |sin(x + iy)| at the witness y = |M| + 1 in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.sin_vertical_unbounded
# description: instances of sin_vertical_unbounded (Contour integration, part I chapter 4), edge case first or marked
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
for xl, x0, Ml, M, e in [('0', 0, '0', 0, True), ('1', 1, '10', 10, False), ('(-5 / 2)', -QQ(5) / 2, '100', 100, False)]:
    instances.append(oracle_instance(xl, Ml, lhs=RI(abs(CB(x0, abs(M) + 1).sin())), rhs=RI(M), relation='>', edge=e))
