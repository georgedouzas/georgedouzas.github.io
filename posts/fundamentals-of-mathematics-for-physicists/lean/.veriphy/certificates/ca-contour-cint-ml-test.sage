# name: ca-contour-cint-ml-test
# technique: test fallback; oracle instances: Sage encloses |integral| and M L(gamma) in ball arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_ML
# description: instances of cint_ML (Contour integration, part I chapter 4), edge case first or marked
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
# f = e^z on the unit circle (M = e), f = z^2 on the segment [0, 1 + i] (M = 2), f = 1 on a point path (edge)
cases = [('(fun w : ℂ => Complex.exp w)', lambda w: w.exp(), '(fun s : ℝ => Complex.exp (s * Complex.I))',
          lambda x: (J * x).exp(), lambda x: J * (J * x).exp(), '0', 0, '(2 * Real.pi)', 2 * CB.pi(), '(Real.exp 1)', CB(1).exp(), False),
         ('(fun w : ℂ => w ^ 2)', lambda w: w ** 2, '(fun s : ℝ => (s : ℂ) * (1 + Complex.I))',
          lambda x: x * (1 + J), lambda x: 1 + J, '0', 0, '1', 1, '2', CB(2), False),
         ('(fun _ : ℂ => 1)', lambda w: CB(1), '(fun _ : ℝ => (0 : ℂ))', lambda x: CB(0), lambda x: CB(0), '0', 0, '1', 1, '1', CB(1), True)]
instances = []
for (fl, F, gl, g, dg, al, a, bl, b, Ml, M, e), Lval in zip(cases, [2 * CB.pi(), CB(2).sqrt(), CB(0)]):
    val = bint(lambda x: F(g(x)) * dg(x), a, b)
    L = Lval
    instances.append(oracle_instance(fl, gl, al, bl, Ml, '(by positivity)', '_', '_', lhs=RI(abs(val)), rhs=RI((M * L).real()), relation='<=', edge=e))
