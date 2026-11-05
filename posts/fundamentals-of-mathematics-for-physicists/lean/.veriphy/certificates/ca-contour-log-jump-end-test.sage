# name: ca-contour-log-jump-end-test
# technique: test fallback; kernel instances: the one-sided limit of the principal logarithm at a = 3, 2, 3/2, from mathlib's limits of log at the cut
# goal: sage_test KaluzaKlein.ComplexAnalysis.CauchyTheorem.log_jump_end
# description: instances of log_jump_end (Contour integration, part I chapter 4), edge case first or marked
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
P = 'have hz : (((1 : ℝ) - AVAL : ℝ) : ℂ) = Complex.exp (((2 * Real.pi : ℝ) : ℂ) * Complex.I) - ((AVAL : ℝ) : ℂ) := by push_cast; rw [Complex.exp_two_pi_mul_I]\nhave hre : (((1 : ℝ) - AVAL : ℝ) : ℂ).re < 0 := by norm_num\nhave him : (((1 : ℝ) - AVAL : ℝ) : ℂ).im = 0 := by simp\nhave hlog := Complex.tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero hre him\nhave hnorm : Real.log ‖(((1 : ℝ) - AVAL : ℝ) : ℂ)‖ = Real.log ((AVAL : ℝ) - 1) := by rw [Complex.norm_real, Real.norm_eq_abs, abs_of_neg (by norm_num)]; ring_nf\nrw [hnorm] at hlog\nrefine hlog.comp (tendsto_nhdsWithin_iff.2 ⟨by rw [hz]; exact ((Complex.continuous_exp.comp (Complex.continuous_ofReal.mul continuous_const)).sub continuous_const).continuousAt.tendsto.mono_left nhdsWithin_le_nhds, ?_⟩)\nfilter_upwards [Ioo_mem_nhdsLT (show Real.pi < 2 * Real.pi by linarith [Real.pi_pos])] with t ht\nsimp only [Set.mem_setOf_eq, Complex.sub_im, Complex.exp_ofReal_mul_I_im, Complex.ofReal_im, sub_zero]\nhave := Real.sin_neg_of_neg_of_neg_pi_lt (show t - 2 * Real.pi < 0 by linarith [ht.2]) (show -Real.pi < t - 2 * Real.pi by linarith [ht.1])\nrwa [Real.sin_sub_two_pi] at this'
instances = [kernel_instance(a, '(by norm_num)', proof=P.replace('AVAL', a), edge=(a == '(3 : ℝ)')) for a in ['(3 : ℝ)', '(2 : ℝ)', '(3 / 2 : ℝ)']]
