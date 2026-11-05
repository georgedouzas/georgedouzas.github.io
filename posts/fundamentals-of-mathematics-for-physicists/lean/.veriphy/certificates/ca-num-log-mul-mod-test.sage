# name: ca-num-log-mul-mod-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_mod file "ca-num-log-mul-mod-test.sage"
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
instances.append(kernel_instance(*['1', '1', 'one_ne_zero', 'one_ne_zero'], proof='exact ⟨0, by simp⟩', edge=True))
instances.append(kernel_instance(*['(-1)', '(-1)', '(by norm_num)', '(by norm_num)'], proof='exact ⟨-1, by rw [show ((-1 : ℂ) * -1) = 1 by norm_num, Complex.log_one, Complex.log_neg_one]; push_cast; ring⟩', edge=False))
instances.append(kernel_instance(*['Complex.I', 'Complex.I', 'Complex.I_ne_zero', 'Complex.I_ne_zero'], proof='exact ⟨0, by rw [Complex.I_mul_I, Complex.log_neg_one, Complex.log_I]; push_cast; ring⟩', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
