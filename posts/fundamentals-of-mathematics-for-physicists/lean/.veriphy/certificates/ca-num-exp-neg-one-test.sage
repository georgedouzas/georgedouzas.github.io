# name: ca-num-exp-neg-one-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_neg_one_iff file "ca-num-exp-neg-one-test.sage"
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
instances.append(kernel_instance(*['0'], proof='constructor\n· intro h\n  norm_num at h\n· rintro ⟨k, hk⟩\n  exfalso\n  have := congrArg Complex.im hk\n  simp at this\n  have h2 : (1 + 2 * (k : ℝ)) * Real.pi = 0 := by linarith\n  rcases mul_eq_zero.mp h2 with h | h\n  · have h3 : (1 + 2 * k : ℤ) = 0 := by exact_mod_cast h\n    omega\n  · exact Real.pi_ne_zero h', edge=True))
instances.append(kernel_instance(*['(Real.pi * Complex.I)'], proof='exact ⟨fun _ => ⟨0, by simp⟩, fun _ => Complex.exp_pi_mul_I⟩', edge=False))
instances.append(kernel_instance(*['(3 * Real.pi * Complex.I)'], proof='constructor\n· intro _\n  exact ⟨1, by push_cast; ring⟩\n· intro _\n  rw [show (3 : ℂ) * Real.pi * Complex.I = Real.pi * Complex.I + 2 * Real.pi * Complex.I by ring,\n    Complex.exp_add, Complex.exp_pi_mul_I, Complex.exp_two_pi_mul_I]\n  ring', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
