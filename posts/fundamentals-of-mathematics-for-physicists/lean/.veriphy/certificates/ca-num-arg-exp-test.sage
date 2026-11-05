# name: ca-num-arg-exp-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.arg_exp_mem file "ca-num-arg-exp-test.sage"
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
instances.append(kernel_instance(*['0', '0'], proof='exact ⟨0, by simp⟩', edge=True))
instances.append(kernel_instance(*['0', '(Real.pi / 2)'], proof='refine ⟨0, ?_⟩\npush_cast\nsimp [Complex.exp_pi_div_two_mul_I]', edge=False))
instances.append(kernel_instance(*['0', 'Real.pi'], proof='refine ⟨0, ?_⟩\npush_cast\nsimp [Complex.exp_pi_mul_I]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
