# name: ca-contour-cint-comp-test
# technique: test fallback; oracle instances: Sage integrates along gamma o phi and along gamma exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourIntegrals.cint_comp
# description: instances of cint_comp (Contour integration, part I chapter 4), edge case first or marked
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
cases = [('(fun w : ℂ => w ^ 2)', z ** 2, '(fun s : ℝ => (s : ℂ) + Complex.I * (s : ℂ) ^ 2)', t + I * t ** 2,
          '(fun s : ℝ => s ^ 2)', '(fun s : ℝ => 2 * s)', t ** 2, '0', 0, '1', 1, False),
         ('(fun w : ℂ => Complex.exp w)', exp(z), '(fun s : ℝ => (s : ℂ) * Complex.I)', t * I,
          '(fun s : ℝ => 3 * s + 1)', '(fun _ : ℝ => 3)', 3 * t + 1, '0', 0, '2', 2, False),
         ('(fun w : ℂ => w)', z, '(fun s : ℝ => (s : ℂ))', t, '(fun s : ℝ => s)', '(fun _ : ℝ => 1)', t, '0', 0, '1', 1, True)]
instances = []
for fl, f, gl, g, pl, dl, phi, cl, c, dl2, d, e in cases:
    lhs = pint(f, g.subs(t=phi), c, d)
    rhs = pint(f, g, phi.subs(t=c), phi.subs(t=d))
    instances.append(oracle_instance(fl, gl, pl, dl, cl, dl2, '_', '_', '_', '_', '_', lhs=ex(lhs), rhs=ex(rhs), edge=e))
