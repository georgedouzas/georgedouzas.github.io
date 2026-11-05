# name: ca-contour-fourier-coeff-test
# technique: test fallback; oracle instances: Sage integrates the trigonometric polynomial against e^(-int) exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ContourWorked.fourier_coeff
# description: instances of fourier_coeff (Contour integration, part I chapter 4), edge case first or marked
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
cases = [(1, '(fun k : ℤ => (k : ℂ) ^ 2)', lambda k: k ** 2, 1, False), (2, '(fun k : ℤ => (k : ℂ) + Complex.I)', lambda k: k + I, -2, False),
         (0, '(fun _ : ℤ => (3 : ℂ))', lambda k: 3, 0, True)]
instances = []
for N, cl, c, n, e in cases:
    p = sum(c(k) * exp(k * t * I) for k in range(-N, N + 1))
    instances.append(oracle_instance(str(N), cl, str(n), '(by decide)', lhs=ex(integrate((p * exp(-n * t * I)).expand(), t, 0, 2 * pi)), rhs=ex(2 * pi * c(n)), edge=e))
