# name: ca-num-disc-open-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.disc_isOpen file "ca-num-disc-open-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='exact isOpen_lt (by fun_prop) continuous_const', edge=True))
instances.append(kernel_instance(*['0', '1'], proof='exact isOpen_lt (by fun_prop) continuous_const', edge=False))
instances.append(kernel_instance(*['Complex.I', '2'], proof='exact isOpen_lt (by fun_prop) continuous_const', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
