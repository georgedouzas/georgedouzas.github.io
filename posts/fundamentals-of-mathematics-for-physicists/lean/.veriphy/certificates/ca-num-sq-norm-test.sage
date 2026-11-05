# name: ca-num-sq-norm-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.sq_norm_eq file "ca-num-sq-norm-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='rw [← Complex.normSq_eq_norm_sq]\nnorm_num [Complex.normSq_apply]', edge=True))
instances.append(kernel_instance(*['3', '4'], proof='rw [← Complex.normSq_eq_norm_sq]\nnorm_num [Complex.normSq_apply]', edge=False))
instances.append(kernel_instance(*['(1 / 2)', '(-2)'], proof='rw [← Complex.normSq_eq_norm_sq]\nnorm_num [Complex.normSq_apply]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
