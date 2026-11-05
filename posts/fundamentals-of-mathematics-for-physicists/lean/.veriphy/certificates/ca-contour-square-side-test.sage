# name: ca-contour-square-side-test
# technique: test fallback; oracle instances: Sage integrates 1/z along the right side of the square exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.square_side
# description: instances of square_side (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('1', 1, True), ('2', 2, False), ('(1 / 3)', QQ(1) / 3, False)]
instances = []
for sl, s, e in cases:
    g = s * (1 - I) + t * (s * (1 + I) - s * (1 - I))
    instances.append(oracle_instance(sl, '(by norm_num)', lhs=ex(integrate((diff(g, t) / g).rectform(), t, 0, 1)), rhs=ex(pi / 2 * I), edge=e))
