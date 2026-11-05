# name: ca-contour-inv-sq-primitive-test
# technique: test fallback; oracle instances: Sage differentiates -1/z exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.inv_sq_primitive
# description: instances of inv_sq_primitive (Contour integration, part I chapter 4), edge case first or marked
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
w = var('w')
instances = [oracle_instance(zl, '(by norm_num)', lhs=ex(diff(-1 / w, w).subs(w=z0)), rhs=ex(1 / z0 ** 2), edge=(z0 == 1))
             for zl, z0 in [('1', 1), ('(1 + Complex.I)', 1 + I), ('(-2 * Complex.I)', -2 * I)]]
