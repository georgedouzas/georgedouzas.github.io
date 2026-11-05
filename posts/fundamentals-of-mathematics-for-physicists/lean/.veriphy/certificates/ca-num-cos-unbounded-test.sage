# name: ca-num-cos-unbounded-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_unbounded file "ca-num-cos-unbounded-test.sage"
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
instances.append(kernel_instance(*['0'], proof='exact ⟨0, by simp⟩', edge=True))
instances.append(kernel_instance(*['(-5)'], proof='exact ⟨0, by norm_num⟩', edge=False))
instances.append(kernel_instance(*['1'], proof='refine ⟨1, ?_⟩\nrw [Complex.cos_mul_I, ← Complex.ofReal_cosh, Complex.norm_real, Real.norm_eq_abs,\n  abs_of_pos (Real.cosh_pos 1)]\nexact Real.one_lt_cosh.mpr one_ne_zero', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
