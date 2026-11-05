# name: ca-num-logs-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logarithms_iff file "ca-num-logs-test.sage"
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
instances.append(kernel_instance(*['1', 'one_ne_zero', '0'], proof='exact ⟨fun _ => ⟨0, by simp⟩, fun _ => by simp⟩', edge=True))
instances.append(kernel_instance(*['(-1)', '(by norm_num)', '(Real.pi * Complex.I)'], proof='exact ⟨fun _ => ⟨0, by simp [Complex.log_neg_one]⟩, fun _ => Complex.exp_pi_mul_I⟩', edge=False))
instances.append(kernel_instance(*['1', 'one_ne_zero', '(2 * Real.pi * Complex.I)'], proof='exact ⟨fun _ => ⟨1, by simp⟩, fun _ => Complex.exp_two_pi_mul_I⟩', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
