# name: ca-num-roots-iff-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.roots_of_unity_iff file "ca-num-roots-iff-test.sage"
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
instances.append(kernel_instance(*['1', 'one_ne_zero', '1'], proof='constructor\n· intro _\n  exact ⟨0, by norm_num, by simp⟩\n· intro _\n  simp', edge=True))
instances.append(kernel_instance(*['2', 'two_ne_zero', '(-1)'], proof='constructor\n· intro _\n  refine ⟨1, by norm_num, ?_⟩\n  push_cast\n  rw [show 2 * (Real.pi : ℂ) * Complex.I * (1 / 2) = Real.pi * Complex.I by ring]\n  exact Complex.exp_pi_mul_I\n· intro _\n  norm_num', edge=False))
instances.append(kernel_instance(*['3', '(by norm_num)', '1'], proof='constructor\n· intro _\n  exact ⟨0, by norm_num, by simp⟩\n· intro _\n  simp', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
