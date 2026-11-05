# name: ca-num-polar-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.polar_form file "ca-num-polar-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0'], proof='simp', edge=True))
instances.append(kernel_instance(*['1'], proof='simp', edge=False))
instances.append(kernel_instance(*['Complex.I'], proof='simp', edge=False))
instances.append(kernel_instance(*['(-1)'], proof='simp', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
