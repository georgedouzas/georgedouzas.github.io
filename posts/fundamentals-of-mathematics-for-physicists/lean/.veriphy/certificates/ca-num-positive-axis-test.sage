# name: ca-num-positive-axis-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.continuous_positive_axis file "ca-num-positive-axis-test.sage"
# description: a continuity statement; each instance is rechecked by the kernel through the general argument; instances with an edge case first
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
instances.append(kernel_instance(*['4', '(by norm_num)'], proof='set x : ℝ := 4 with hxv\nhave hx : 0 < x := by rw [hxv]; norm_num\nhave hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx\nexact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩', edge=True))
instances.append(kernel_instance(*['1', '(by norm_num)'], proof='set x : ℝ := 1 with hxv\nhave hx : 0 < x := by rw [hxv]; norm_num\nhave hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx\nexact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩', edge=False))
instances.append(kernel_instance(*['(1 / 2)', '(by norm_num)'], proof='set x : ℝ := (1 / 2) with hxv\nhave hx : 0 < x := by rw [hxv]; norm_num\nhave hs : (x : ℂ) ∈ slitPlane := by simpa [mem_slitPlane_iff] using hx\nexact ⟨continuousAt_clog hs, continuousAt_cpow_const hs⟩', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
