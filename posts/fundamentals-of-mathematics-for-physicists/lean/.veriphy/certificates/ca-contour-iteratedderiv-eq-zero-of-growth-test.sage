# name: ca-contour-iteratedderiv-eq-zero-of-growth-test
# technique: test fallback; oracle instances: Sage differentiates polynomials of degree at most n exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.iteratedDeriv_eq_zero_of_growth
# description: instances of iteratedDeriv_eq_zero_of_growth (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun u : ℂ => u ^ 2)', u ** 2, 2, 1, 3, False), ('(fun u : ℂ => u ^ 2 + 3 * u + 1)', u ** 2 + 3 * u + 1, 2, 5, 4, False),
         ('(fun _ : ℂ => (2 : ℂ))', SR(2), 0, 2, 1, True)]
instances = [oracle_instance(fl, '(by fun_prop)', str(n), str(C), '_', str(k), '(by norm_num)', lhs=SR(diff(f, u, k).subs(u=0)), rhs=SR(0), edge=e)
             for fl, f, n, C, k, e in cases]
