# name: ca-num-log-jump-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.log_jump file "ca-num-log-jump-test.sage"
# description: a limit statement; each instance is rechecked by the kernel through the general argument; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
CI = ComplexIntervalField(128)
def cdiff(a, b):
    d = CI(a) - CI(b)
    return RI(d.real().abs() + d.imag().abs())
CB = ComplexBallField(128)
def bdiff(a, b):
    d = a - b
    return RI(d.real().abs()) + RI(d.imag().abs())
instances.append(kernel_instance(*['(-4)', '(by norm_num)'], proof='set x : ℝ := (-4) with hxv\nhave hx : x < 0 := by rw [hxv]; norm_num\nconstructor\n· have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)\n  have hv : Complex.log x = Real.log |x| + π * I := by\n    apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]\n  rw [ContinuousWithinAt, hv] at h\n  exact h\n· have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)\n  simpa using h', edge=True))
instances.append(kernel_instance(*['(-1)', '(by norm_num)'], proof='set x : ℝ := (-1) with hxv\nhave hx : x < 0 := by rw [hxv]; norm_num\nconstructor\n· have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)\n  have hv : Complex.log x = Real.log |x| + π * I := by\n    apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]\n  rw [ContinuousWithinAt, hv] at h\n  exact h\n· have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)\n  simpa using h', edge=False))
instances.append(kernel_instance(*['(-1 / 2)', '(by norm_num)'], proof='set x : ℝ := (-1 / 2) with hxv\nhave hx : x < 0 := by rw [hxv]; norm_num\nconstructor\n· have h := continuousWithinAt_log_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)\n  have hv : Complex.log x = Real.log |x| + π * I := by\n    apply Complex.ext <;> simp [Complex.log_re, Complex.log_im, arg_ofReal_of_neg hx]\n  rw [ContinuousWithinAt, hv] at h\n  exact h\n· have h := tendsto_log_nhdsWithin_im_neg_of_re_neg_of_im_zero (z := (x : ℂ)) (by simpa using hx) (by simp)\n  simpa using h', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
