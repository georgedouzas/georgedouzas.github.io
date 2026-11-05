# name: ca-num-arg-above-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_above file "ca-num-arg-above-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['(-1)', '(by norm_num)', '(by simp)'], proof='have h := Complex.continuousWithinAt_arg_of_re_neg_of_im_zero (z := (-1 : ℂ)) (by norm_num) (by simp)\nrwa [ContinuousWithinAt, Complex.arg_eq_pi_iff.mpr ⟨by norm_num, by simp⟩] at h', edge=True))
instances.append(kernel_instance(*['(-4)', '(by norm_num)', '(by simp)'], proof='have h := Complex.continuousWithinAt_arg_of_re_neg_of_im_zero (z := (-4 : ℂ)) (by norm_num) (by simp)\nrwa [ContinuousWithinAt, Complex.arg_eq_pi_iff.mpr ⟨by norm_num, by simp⟩] at h', edge=False))
instances.append(kernel_instance(*['(-1 / 2)', '(by norm_num)', '(by simp)'], proof='have h := Complex.continuousWithinAt_arg_of_re_neg_of_im_zero (z := (-1 / 2 : ℂ)) (by norm_num) (by simp)\nrwa [ContinuousWithinAt, Complex.arg_eq_pi_iff.mpr ⟨by norm_num, by simp⟩] at h', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
