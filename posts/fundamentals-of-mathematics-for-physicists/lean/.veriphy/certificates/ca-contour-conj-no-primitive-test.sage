# name: ca-contour-conj-no-primitive-test
# technique: test fallback; oracle instances: at a point p, Sage shows F'(p) differs from conj(p), so F is not a primitive of conj
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.conj_no_primitive
# description: instances of conj_no_primitive (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun w : ℂ => w ^ 2 / 2)', w ** 2 / 2, I, False), ('(fun _ : ℂ => (0 : ℂ))', SR(0), 1, True), ('Complex.exp', exp(w), 0, False)]
instances = [oracle_instance(Fl, lhs=ex(abs(diff(F, w).subs(w=p) - conjugate(p))), rhs=SR(0), relation='>', edge=e)
             for Fl, F, p, e in cases]
