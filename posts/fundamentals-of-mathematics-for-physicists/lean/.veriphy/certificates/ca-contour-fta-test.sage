# name: ca-contour-fta-test
# technique: test fallback; kernel instances: explicit roots, checked by simp
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.fta
# description: instances of fta (Contour integration, part I chapter 4), edge case first or marked
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
instances = [kernel_instance('(Polynomial.X : Polynomial ℂ)', '(by rw [Polynomial.natDegree_X]; norm_num)', proof='exact ⟨0, by simp⟩', edge=True),
             kernel_instance('(Polynomial.X ^ 2 + 1 : Polynomial ℂ)', '(by rw [show (Polynomial.X ^ 2 + 1 : Polynomial ℂ).natDegree = 2 from by compute_degree!]; norm_num)', proof='exact ⟨Complex.I, by simp⟩'),
             kernel_instance('(Polynomial.X - 1 : Polynomial ℂ)', '(by rw [show (Polynomial.X - 1 : Polynomial ℂ).natDegree = 1 from by compute_degree!]; norm_num)', proof='exact ⟨1, by simp⟩')]
