# name: ca-contour-cauchy-estimate-test
# technique: test fallback; oracle instances: Sage computes |f^(n)(c)| exactly and compares it with n! M / R^n in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.cauchy_estimate
# description: instances of cauchy_estimate (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('Complex.exp', exp(u), '0', 0, '1', 1, '(Real.exp 1)', RI(1).exp(), 0, True), ('Complex.exp', exp(u), '0', 0, '1', 1, '(Real.exp 1)', RI(1).exp(), 10, False),
         ('(fun v : ℂ => v ^ 3)', u ** 3, '0', 0, '2', 2, '8', RI(8), 3, False), ('Complex.exp', exp(u), '0', 0, '3', 3, '(Real.exp 3)', RI(3).exp(), 5, False)]
instances = [oracle_instance(fl, cl, Rl, Ml, str(n), '(by norm_num)', '_', '_', lhs=RI(abs(CB(diff(f, u, n).subs(u=c)))), rhs=factorial(n) * M / RI(R) ** n, relation='<=', edge=e)
             for fl, f, cl, c, Rl, R, Ml, M, n, e in cases]
