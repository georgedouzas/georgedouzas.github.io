# name: ca-num-sqrt-roots-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.two_square_roots file "ca-num-sqrt-roots-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', 'one_ne_zero'], proof='refine ⟨1, by norm_num, by norm_num, fun u hu => ?_⟩\nhave h : (u - 1) * (u + 1) = 0 := by linear_combination hu\nrcases mul_eq_zero.mp h with h | h\n· exact Or.inl (by linear_combination h)\n· exact Or.inr (by linear_combination h)', edge=True))
instances.append(kernel_instance(*['(-4)', '(by norm_num)'], proof='refine ⟨2 * Complex.I, by linear_combination 4 * Complex.I_sq, fun h => by have := congrArg Complex.im h; norm_num at this, fun u hu => ?_⟩\nhave h : (u - 2 * Complex.I) * (u + 2 * Complex.I) = 0 := by linear_combination hu - 4 * Complex.I_sq\nrcases mul_eq_zero.mp h with h | h\n· exact Or.inl (by linear_combination h)\n· exact Or.inr (by linear_combination h)', edge=False))
instances.append(kernel_instance(*['(2 * Complex.I)', '(by simp)'], proof='refine ⟨1 + Complex.I, by linear_combination Complex.I_sq, fun h => by have := congrArg Complex.re h; norm_num at this, fun u hu => ?_⟩\nhave h : (u - (1 + Complex.I)) * (u + (1 + Complex.I)) = 0 := by linear_combination hu - Complex.I_sq\nrcases mul_eq_zero.mp h with h | h\n· exact Or.inl (by linear_combination h)\n· exact Or.inr (by linear_combination h)', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
