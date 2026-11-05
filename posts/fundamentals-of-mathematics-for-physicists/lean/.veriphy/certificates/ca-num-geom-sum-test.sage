# name: ca-num-geom-sum-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.NumbersWorked.geom_sum_identity file "ca-num-geom-sum-test.sage"
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
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(kernel_instance(*['Complex.I', '3'], proof='simp only [Finset.sum_range_succ, Finset.sum_range_zero]\nring', edge=False))
instances.append(kernel_instance(*['2', '4'], proof='simp only [Finset.sum_range_succ, Finset.sum_range_zero]\nring', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
