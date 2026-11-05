# name: ca-num-lower-bound-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.lower_bound_on_circle file "ca-num-lower-bound-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['2', '(by norm_num)'], proof='norm_num', edge=True))
instances.append(kernel_instance(*['(2 * Complex.I)', '(by simp)'], proof='rw [show (2 * Complex.I) ^ 2 + 1 = (-3 : ℂ) by linear_combination 4 * Complex.I_sq]\nnorm_num', edge=False))
instances.append(kernel_instance(*['(-2)', '(by norm_num)'], proof='norm_num', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
