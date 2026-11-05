# name: ca-num-dist-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.dist_eq_norm_sub file "ca-num-dist-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(kernel_instance(*['1', 'Complex.I'], proof='exact dist_eq_norm _ _', edge=False))
instances.append(kernel_instance(*['(2 + Complex.I)', '3'], proof='exact dist_eq_norm _ _', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
