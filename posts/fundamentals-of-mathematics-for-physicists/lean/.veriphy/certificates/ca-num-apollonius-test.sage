# name: ca-num-apollonius-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.apollonius file "ca-num-apollonius-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='norm_num', edge=True))
instances.append(kernel_instance(*['(-3)', '0'], proof='norm_num', edge=False))
instances.append(kernel_instance(*['5', '0'], proof='norm_num', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
