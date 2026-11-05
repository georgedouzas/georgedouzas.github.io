# name: ca-contour-circle-separates-test
# technique: test fallback; kernel instances: the complement of the circle at explicit centres and radii, from mathlib's balls
# goal: sage_test KaluzaKlein.ComplexAnalysis.Contours.circle_separates
# description: instances of circle_separates (Contour integration, part I chapter 4), edge case first or marked
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
P = """refine ⟨?_, Metric.isBounded_ball, fun h => NormedSpace.unbounded_univ ℝ ℂ ?_⟩
· ext z
  simp only [Set.mem_compl_iff, Metric.mem_sphere, Set.mem_union, Metric.mem_ball, Metric.mem_closedBall, not_le]
  exact ⟨fun h => lt_or_gt_of_ne h, fun h => h.elim ne_of_lt ne_of_gt⟩
· rw [← Set.union_compl_self (Metric.closedBall _ _)]
  exact Metric.isBounded_closedBall.union h"""
instances = [kernel_instance(c, r, proof=P, edge=(r == '0')) for c, r in [('0', '0'), ('0', '1'), ('(1 + Complex.I)', '(5 / 2)')]]
