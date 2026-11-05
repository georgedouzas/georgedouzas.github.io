# name: ca-num-nth-root-distinct-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values_distinct file "ca-num-nth-root-distinct-test.sage"
# description: the claim is an injectivity statement; its hypothesis holds only at instances with j = k; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
CI = ComplexIntervalField(128)
def cdiff(a, b):
    d = CI(a) - CI(b)
    return RI(d.real().abs() + d.imag().abs())
CB = ComplexBallField(128)
def bdiff(a, b):
    d = a - b
    return RI(d.real().abs()) + RI(d.imag().abs())
instances.append(kernel_instance(*['1', '1', 'one_ne_zero', '0', '0', 'one_pos', 'one_pos', 'rfl'], proof='rfl', edge=True))
instances.append(kernel_instance(*['Complex.I', '3', '(by norm_num)', '2', '2', '(by norm_num)', '(by norm_num)', 'rfl'], proof='rfl', edge=False))
instances.append(kernel_instance(*['(-8)', '3', '(by norm_num)', '1', '1', '(by norm_num)', '(by norm_num)', 'rfl'], proof='rfl', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
