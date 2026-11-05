# name: ca-contour-circle-deriv-test
# technique: test fallback; oracle instances: Sage differentiates z0 + r e^(it) exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Contours.circle_deriv
# description: instances of circle_deriv (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('0', 0, '1', 1, '0', 0, True), ('(1 + 2 * Complex.I)', 1 + 2 * I, '3', 3, '(Real.pi / 3)', pi / 3, False),
         ('(-Complex.I)', -I, '(1 / 2)', QQ(1) / 2, '2', 2, False)]
instances = [oracle_instance(zl, rl, tl, lhs=ex(diff(z0 + r * exp(I * t), t).subs(t=t0)), rhs=ex(I * r * exp(t0 * I)), edge=e)
             for zl, z0, rl, r, tl, t0, e in cases]
