# name: ca-num-triangle-eq-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.triangle_eq_iff file "ca-num-triangle-eq-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '1'], proof='simp', edge=True))
instances.append(kernel_instance(*['1', '1'], proof='norm_num', edge=False))
instances.append(kernel_instance(*['1', '(-1)'], proof='norm_num [Complex.arg_neg_one, Real.pi_pos.ne]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
