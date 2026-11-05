# name: ca-num-fout-branch-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.fOut_branch_at file "ca-num-fout-branch-test.sage"
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
instances.append(kernel_instance(*['2', '(by norm_num)'], proof='have hz0 : (2 : ℂ) ≠ 0 := by norm_num\nrefine ⟨?_, fOut_sq _ hz0⟩\nexact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit _ (by norm_num))).comp\n  (f := fun w : ℂ => 1 - 1 / w ^ 2) (x := _) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))', edge=True))
instances.append(kernel_instance(*['(-3)', '(by norm_num)'], proof='have hz0 : (-3 : ℂ) ≠ 0 := by norm_num\nrefine ⟨?_, fOut_sq _ hz0⟩\nexact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit _ (by norm_num))).comp\n  (f := fun w : ℂ => 1 - 1 / w ^ 2) (x := _) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))', edge=False))
instances.append(kernel_instance(*['(2 * Complex.I)', '(by simp)'], proof='have hz0 : (2 * Complex.I : ℂ) ≠ 0 := by simp\nrefine ⟨?_, fOut_sq _ hz0⟩\nexact continuousAt_id.mul ((continuousAt_cpow_const (fOut_slit _ (by simp))).comp\n  (f := fun w : ℂ => 1 - 1 / w ^ 2) (x := _) (by fun_prop (disch := exact pow_ne_zero 2 hz0)))', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
