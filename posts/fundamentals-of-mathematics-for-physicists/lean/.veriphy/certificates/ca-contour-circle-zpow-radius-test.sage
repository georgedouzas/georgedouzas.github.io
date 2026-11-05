# name: ca-contour-circle-zpow-radius-test
# technique: test fallback; oracle instances: Sage integrates both sides exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourWorked.circle_zpow_radius
# description: instances of circle_zpow_radius (Contour integration, part I chapter 4), edge case first or marked
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
cases = [(-1, '1', 1, True), (2, '2', 2, False), (-3, '(1 / 2)', QQ(1) / 2, False), (0, '3', 3, False)]
instances = [oracle_instance(str(n), rl, '(by norm_num)', lhs=ex(pint(z ** n, r * exp(I * t), 0, 2 * pi)),
             rhs=ex(I * r ** (n + 1) * integrate(exp((n + 1) * t * I), t, 0, 2 * pi)), edge=e) for n, rl, r, e in cases]
