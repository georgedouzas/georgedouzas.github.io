# name: ca-contour-inv-poly-tendsto-zero-test
# technique: test fallback; kernel instances: |p| tends to infinity, so 1/p tends to 0
# goal: sage_test KaluzaKlein.ComplexAnalysis.LiouvilleTheorem.inv_poly_tendsto_zero
# description: instances of inv_poly_tendsto_zero (Contour integration, part I chapter 4), edge case first or marked
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
P = 'have hd : 0 < (PVAL).degree := Polynomial.natDegree_pos_iff_degree_pos.1 HP\nrw [tendsto_zero_iff_norm_tendsto_zero]\nsimpa [norm_inv, Function.comp_def] using tendsto_inv_atTop_zero.comp (Polynomial.tendsto_norm_atTop PVAL hd tendsto_norm_cocompact_atTop)'
cases = [('(Polynomial.X : Polynomial ℂ)', '(by rw [Polynomial.natDegree_X]; norm_num)', True),
         ('(Polynomial.X ^ 2 : Polynomial ℂ)', '(by rw [Polynomial.natDegree_X_pow]; norm_num)', False),
         ('(Polynomial.X ^ 3 : Polynomial ℂ)', '(by rw [Polynomial.natDegree_X_pow]; norm_num)', False)]
instances = [kernel_instance(p, hp, proof=P.replace('PVAL', p).replace('HP', hp), edge=e) for p, hp, e in cases]
