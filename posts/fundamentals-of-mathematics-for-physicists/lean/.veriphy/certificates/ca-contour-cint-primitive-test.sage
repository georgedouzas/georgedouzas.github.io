# name: ca-contour-cint-primitive-test
# technique: test fallback; oracle instances: Sage integrates f along the path and evaluates F at the endpoints exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_primitive
# description: instances of cint_primitive (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun w : ℂ => w ^ 2)', z ** 2, '(fun w : ℂ => w ^ 3 / 3)', z ** 3 / 3, '(fun s : ℝ => (s : ℂ) + Complex.I * (s : ℂ) ^ 2)', t + I * t ** 2, '0', 0, '1', 1, False),
         ('Complex.exp', exp(z), 'Complex.exp', exp(z), '(fun s : ℝ => (s : ℂ) * Complex.I)', t * I, '0', 0, 'Real.pi', pi, False),
         ('(fun w : ℂ => w)', z, '(fun w : ℂ => w ^ 2 / 2)', z ** 2 / 2, '(fun s : ℝ => (s : ℂ))', t, '2', 2, '2', 2, True)]
instances = [oracle_instance(fl, Fl, gl, al, bl, '_', '_', '_', lhs=ex(pint(f, g, a, b)), rhs=ex(F.subs(z=g.subs(t=b)) - F.subs(z=g.subs(t=a))), edge=e)
             for fl, f, Fl, F, gl, g, al, a, bl, b, e in cases]
