# name: ca-contour-const-of-sqrt-growth-test
# technique: test fallback; kernel instances: the only entire functions bounded by C sqrt|z| (zero at 0) are 0
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.const_of_sqrt_growth
# description: instances of const_of_sqrt_growth (Contour integration, part I chapter 4), edge case first or marked
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
instances = [kernel_instance('(fun _ : ℂ => (0 : ℂ))', '(differentiable_const _)', C, '(fun _ => by simp only [norm_zero]; positivity)', z, w, proof='rfl', edge=(C == '0'))
             for C, z, w in [('0', '0', '1'), ('1', 'Complex.I', '(2 + Complex.I)'), ('5', '(-3)', '(7 * Complex.I)')]]
