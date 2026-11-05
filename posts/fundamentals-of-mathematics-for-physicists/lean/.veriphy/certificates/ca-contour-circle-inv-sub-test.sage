# name: ca-contour-circle-inv-sub-test
# technique: test fallback; oracle instances: Sage encloses the circle integral of 1/(z - w) in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.circle_inv_sub
# description: instances of circle_inv_sub (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('3', 3, '1', 1, False), ('3', 3, '4', 4, False), ('(1 + Complex.I)', 1 + I, '2', 2, False), ('0', 0, '1', 1, True)]
instances = []
for wl, w0, Rl, R, e in cases:
    val = circ(lambda w: 1 / (w - CB(w0)), 0, R)
    expected = 2 * CB.pi() * J if abs(CC(w0)) < R else CB(0)
    instances.append(oracle_instance(wl, Rl, '(by norm_num)', '(by norm_num)', lhs=gap((val, expected)), rhs=RI(0), edge=e))
