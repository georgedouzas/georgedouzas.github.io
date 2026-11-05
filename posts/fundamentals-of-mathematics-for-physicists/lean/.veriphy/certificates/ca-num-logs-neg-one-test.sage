# name: ca-num-logs-neg-one-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.logs_neg_one file "ca-num-logs-neg-one-test.sage"
# description: no recipe certifies the general statement; w = 0 (edge, both sides false), iπ (k = 0), -iπ (k = -1), 3iπ (k = 1)
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
instances.append(kernel_instance(*['0'], proof='constructor\n· intro h\n  norm_num at h\n· rintro ⟨k, hk⟩\n  exfalso\n  have := congrArg Complex.im hk\n  simp at this\n  have h3 : ((2 * k + 1 : ℤ) : ℝ) = 0 := by push_cast; linarith\n  have h4 : (2 * k + 1 : ℤ) = 0 := by exact_mod_cast h3\n  omega', edge=True))
instances.append(kernel_instance(*['(Real.pi * Complex.I)'], proof='exact ⟨fun _ => ⟨0, by simp⟩, fun _ => Complex.exp_pi_mul_I⟩', edge=True))
instances.append(kernel_instance(*['(-(Real.pi * Complex.I))'], proof='constructor\n· intro _\n  exact ⟨-1, by push_cast; ring⟩\n· intro _\n  rw [Complex.exp_neg, Complex.exp_pi_mul_I]\n  norm_num', edge=False))
instances.append(kernel_instance(*['(3 * Real.pi * Complex.I)'], proof='constructor\n· intro _\n  exact ⟨1, by push_cast; ring⟩\n· intro _\n  rw [show (3 * Real.pi * Complex.I : ℂ) = Real.pi * Complex.I + (1 : ℤ) * (2 * Real.pi * Complex.I) by push_cast; ring, Complex.exp_add, Complex.exp_int_mul_two_pi_mul_I, Complex.exp_pi_mul_I]\n  ring', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
