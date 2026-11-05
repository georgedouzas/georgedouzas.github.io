# name: ca-num-root-branch-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.root_branch file "ca-num-root-branch-test.sage"
# description: a continuity and branch statement; each instance is rechecked by the kernel through the general argument; instances with an edge case first
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
instances.append(kernel_instance(*['1', 'one_ne_zero'], proof='refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩\nshow (z ^ (1 / _ : ℂ)) ^ _ = z\nrw [one_div]\nexact Complex.cpow_nat_inv_pow z (by norm_num)', edge=True))
instances.append(kernel_instance(*['2', 'two_ne_zero'], proof='refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩\nshow (z ^ (1 / _ : ℂ)) ^ _ = z\nrw [one_div]\nexact Complex.cpow_nat_inv_pow z (by norm_num)', edge=False))
instances.append(kernel_instance(*['3', '(by norm_num)'], proof='refine ⟨⟨fun z hz => (continuousAt_cpow_const hz).continuousWithinAt, fun z _ => ?_⟩, Complex.one_cpow _⟩\nshow (z ^ (1 / _ : ℂ)) ^ _ = z\nrw [one_div]\nexact Complex.cpow_nat_inv_pow z (by norm_num)', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
