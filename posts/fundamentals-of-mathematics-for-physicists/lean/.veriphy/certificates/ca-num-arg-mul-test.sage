# name: ca-num-arg-mul-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arg_mul_mod file "ca-num-arg-mul-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', '1', 'one_ne_zero', 'one_ne_zero'], proof='simp', edge=True))
instances.append(kernel_instance(*['Complex.I', 'Complex.I', 'Complex.I_ne_zero', 'Complex.I_ne_zero'], proof='rw [Complex.I_mul_I, Complex.arg_neg_one, Complex.arg_I, ← Real.Angle.coe_add]\ncongr 1\nring', edge=False))
instances.append(kernel_instance(*['(-1)', '(-1)', '(by norm_num)', '(by norm_num)'], proof='rw [show ((-1 : ℂ) * -1) = 1 by norm_num, Complex.arg_one, Complex.arg_neg_one, ← Real.Angle.coe_add,\n  show Real.pi + Real.pi = 2 * Real.pi by ring, Real.Angle.coe_two_pi]\nsimp', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
