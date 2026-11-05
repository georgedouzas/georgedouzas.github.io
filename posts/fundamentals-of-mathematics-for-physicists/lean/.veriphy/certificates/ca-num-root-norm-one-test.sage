# name: ca-num-root-norm-one-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_norm file "ca-num-root-norm-one-test.sage"
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
instances.append(kernel_instance(*['0', '0'], proof='rw [Complex.norm_exp]\nsimp', edge=True))
instances.append(kernel_instance(*['6', '1'], proof='rw [Complex.norm_exp]\nsimp', edge=False))
instances.append(kernel_instance(*['4', '3'], proof='rw [Complex.norm_exp]\nsimp', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
