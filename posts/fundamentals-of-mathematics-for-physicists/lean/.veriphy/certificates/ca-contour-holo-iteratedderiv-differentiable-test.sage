# name: ca-contour-holo-iteratedderiv-differentiable-test
# technique: test fallback; kernel instances: the n-th derivative of exp is exp, differentiable everywhere
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyIntegralFormula.holo_iteratedDeriv_differentiable
# description: instances of holo_iteratedDeriv_differentiable (Contour integration, part I chapter 4), edge case first or marked
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
P = 'rw [iteratedDeriv_eq_iterate, Complex.iter_deriv_exp]\nexact Complex.differentiable_exp.differentiableOn'
instances = [kernel_instance('Complex.exp', U, hU, '(Complex.differentiable_exp.differentiableOn)', str(n), proof=P, edge=(n == 0))
             for U, hU, n in [('Set.univ', 'isOpen_univ', 0), ('Set.univ', 'isOpen_univ', 3), ('(Metric.ball 0 1)', 'Metric.isOpen_ball', 1)]]
