# name: ca-contour-derivs-from-boundary-test
# technique: test fallback; oracle instances: Sage differentiates two expressions of the same function exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.derivs_from_boundary
# description: instances of derivs_from_boundary (Contour integration, part I chapter 4), edge case first or marked
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
u = var('u')
cases = [('(fun u : ℂ => (u + 1) ^ 2)', (u + 1) ** 2, '(fun u : ℂ => u ^ 2 + 2 * u + 1)', u ** 2 + 2 * u + 1, '0', 0, 1, False),
         ('(fun u : ℂ => Complex.sin (2 * u))', sin(2 * u), '(fun u : ℂ => 2 * Complex.sin u * Complex.cos u)', 2 * sin(u) * cos(u), '(1 + Complex.I)', 1 + I, 3, False),
         ('Complex.exp', exp(u), 'Complex.exp', exp(u), '0', 0, 0, True)]
instances = [oracle_instance(fl, gl, cl, '1', str(n), '(by norm_num)', '_', '_', '_', lhs=gap((CB(diff(f, u, n).subs(u=c)), CB(diff(g, u, n).subs(u=c)))), rhs=RI(0), edge=e)
             for fl, f, gl, g, cl, c, n, e in cases]
