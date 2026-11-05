# name: ca-num-mul-inv-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.mul_inv_of_ne_zero file "ca-num-mul-inv-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', 'one_ne_zero'], proof='exact ⟨1, by simp⟩', edge=True))
instances.append(kernel_instance(*['Complex.I', 'Complex.I_ne_zero'], proof='exact ⟨-Complex.I, by linear_combination -Complex.I_sq⟩', edge=False))
instances.append(kernel_instance(*['(2 + Complex.I)', '(by intro h; have := congrArg Complex.re h; norm_num at this)'], proof='exact ⟨(2 - Complex.I) / 5, by linear_combination -(1 / 5 : ℂ) * Complex.I_sq⟩', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
