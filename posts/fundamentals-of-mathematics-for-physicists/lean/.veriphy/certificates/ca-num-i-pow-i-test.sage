# name: ca-num-i-pow-i-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.I_pow_I_values file "ca-num-i-pow-i-test.sage"
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
instances.append(kernel_instance(*['0'], proof='push_cast\ncongr 1\nlinear_combination ((Real.pi : ℂ) / 2 + 2 * (Real.pi : ℂ) * (0)) * Complex.I_sq', edge=True))
instances.append(kernel_instance(*['1'], proof='push_cast\ncongr 1\nlinear_combination ((Real.pi : ℂ) / 2 + 2 * (Real.pi : ℂ) * (1)) * Complex.I_sq', edge=False))
instances.append(kernel_instance(*['(-2)'], proof='push_cast\ncongr 1\nlinear_combination ((Real.pi : ℂ) / 2 + 2 * (Real.pi : ℂ) * (-2)) * Complex.I_sq', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
