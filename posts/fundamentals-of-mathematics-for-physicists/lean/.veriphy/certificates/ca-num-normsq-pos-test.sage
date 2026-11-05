# name: ca-num-normsq-pos-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.normSq_pos_of_ne_zero file "ca-num-normsq-pos-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', 'one_ne_zero'], proof='norm_num', edge=True))
instances.append(kernel_instance(*['Complex.I', 'Complex.I_ne_zero'], proof='norm_num', edge=False))
instances.append(kernel_instance(*['(2 + Complex.I)', '(by intro h; have := congrArg Complex.re h; norm_num at this)'], proof='norm_num', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
