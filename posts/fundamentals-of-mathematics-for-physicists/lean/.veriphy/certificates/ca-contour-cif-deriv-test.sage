# name: ca-contour-cif-deriv-test
# technique: test fallback; oracle instances: Sage differentiates f exactly and encloses n!/(2 pi i) times the circle integral in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cif_deriv
# description: instances of cif_deriv (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('Complex.exp', exp(u), lambda v: v.exp(), '0', 0, '1', 1, 0, True), ('Complex.exp', exp(u), lambda v: v.exp(), '0', 0, '1', 1, 3, False),
         ('Complex.sin', sin(u), lambda v: v.sin(), '(1 + Complex.I)', 1 + I, '2', 2, 2, False),
         ('(fun v : ℂ => v ^ 4)', u ** 4, lambda v: v ** 4, '1', 1, '(1 / 2)', QQ(1) / 2, 4, False)]
instances = [oracle_instance(fl, cl, Rl, str(n), '(by norm_num)', '_', lhs=gap((2 * CB.pi() * J * CB(diff(f, u, n).subs(u=c)), factorial(n) * circ(lambda v: F(v) / (v - CB(c)) ** (n + 1), c, R))), rhs=RI(0), edge=e)
             for fl, f, F, cl, c, Rl, R, n, e in cases]
