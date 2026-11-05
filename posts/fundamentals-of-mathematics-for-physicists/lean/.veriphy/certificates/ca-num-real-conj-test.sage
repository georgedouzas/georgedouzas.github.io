# name: ca-num-real-conj-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.real_iff_conj file "ca-num-real-conj-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0'], proof='simp [Complex.ext_iff]', edge=True))
instances.append(kernel_instance(*['3'], proof='simp [Complex.ext_iff]', edge=False))
instances.append(kernel_instance(*['(2 + Complex.I)'], proof='simp [Complex.ext_iff]\nall_goals norm_num', edge=False))
instances.append(kernel_instance(*['Complex.I'], proof='simp [Complex.ext_iff]\nall_goals norm_num', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
