# name: ca-num-modulus-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.modulus_formula file "ca-num-modulus-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='rw [Complex.norm_def, Complex.normSq_apply]\nnorm_num', edge=True))
instances.append(kernel_instance(*['3', '4'], proof='rw [Complex.norm_def, Complex.normSq_apply]\nnorm_num', edge=False))
instances.append(kernel_instance(*['1', '(-2)'], proof='rw [Complex.norm_def, Complex.normSq_apply]\nnorm_num', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
