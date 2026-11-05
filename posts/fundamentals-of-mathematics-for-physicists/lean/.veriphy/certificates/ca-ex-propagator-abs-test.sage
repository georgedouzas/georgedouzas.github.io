# name: ca-ex-propagator-abs-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Exercises.propagator_abs file "ca-ex-propagator-abs-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
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
instances.append(kernel_instance(*['0', '0'], proof='constructor <;> intro h <;> simp at h', edge=True))
instances.append(kernel_instance(*['1', '1'], proof='constructor\n· intro _\n  simp\n· intro h\n  norm_num at h', edge=False))
instances.append(kernel_instance(*['1', '(-1)'], proof='constructor\n· intro h\n  norm_num at h\n· intro _\n  norm_num', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
