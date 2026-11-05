# name: ca-contour-mean-value-re-test
# technique: test fallback; oracle instances: Sage encloses the average of Re f over the circle in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.mean_value_re
# description: instances of mean_value_re (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('Complex.exp', lambda u: u.exp(), '0', 0, '1', 1, True), ('(fun u : ℂ => u ^ 2 + 3 * u)', lambda u: u ** 2 + 3 * u, '(1 + Complex.I)', 1 + I, '2', 2, False),
         ('Complex.sin', lambda u: u.sin(), '(-1)', -1, '(1 / 2)', QQ(1) / 2, False)]
instances = [oracle_instance(fl, cl, rl, '(by norm_num)', '_', lhs=gap((F(CB(c)).real(), (bint(lambda x: F(CB(c) + CB(r) * (J * x).exp()), 0, 2 * CB.pi()) / (2 * CB.pi())).real())), rhs=RI(0), edge=e)
             for fl, F, cl, c, rl, r, e in cases]
