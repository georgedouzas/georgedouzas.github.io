# name: ca-contour-exp-norm-le-on-circle-test
# technique: test fallback; oracle instances: Sage compares |e^z| with e at points of the unit circle in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.exp_norm_le_on_circle
# description: instances of exp_norm_le_on_circle (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('1', 1, True), ('Complex.I', I, False), ('(3 / 5 + 4 / 5 * Complex.I)', QQ(3) / 5 + QQ(4) / 5 * I, False), ('(-1)', -1, False)]
instances = [oracle_instance(zl, '_', lhs=exp(real_part(z0)), rhs=e, relation='<=', edge=ed) for zl, z0, ed in cases]
