# name: ca-contour-cauchy-annulus-test
# technique: test fallback; oracle instances: Sage encloses both circle integrals in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.cauchy_annulus
# description: instances of cauchy_annulus (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun w : ℂ => w⁻¹)', lambda w: 1 / w, '1', 1, '2', 2, False), ('(fun w : ℂ => (w ^ 2)⁻¹)', lambda w: 1 / w ** 2, '1', 1, '3', 3, False),
         ('(fun w : ℂ => Complex.exp w / w)', lambda w: w.exp() / w, '(1 / 2)', QQ(1) / 2, '(1 / 2)', QQ(1) / 2, True)]
instances = [oracle_instance(fl, '0', rl, Rl, '(by norm_num)', '(by norm_num)', '_', lhs=gap((circ(F, 0, R), circ(F, 0, r))), rhs=RI(0), edge=e)
             for fl, F, rl, r, Rl, R, e in cases]
