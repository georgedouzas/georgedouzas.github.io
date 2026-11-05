# name: ca-num-dot-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_dot file "ca-num-dot-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0', '0', '0'], proof='simp', edge=True))
instances.append(kernel_instance(*['1', '2', '3', '4'], proof='simp [Complex.mul_re]\nall_goals ring', edge=False))
instances.append(kernel_instance(*['(1 / 2)', '(-1)', '2', '3'], proof='simp [Complex.mul_re]\nall_goals ring', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
