# name: ca-num-exp-one-add-I-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_eq_one_add_I_iff file "ca-num-exp-one-add-I-test.sage"
# description: no recipe certifies the general statement; z = 0 (edge), 1, 2 (both sides false: imaginary parts) and the solution k = 0 (Sage, 128-bit intervals)
SEED = 0
RI = RealIntervalField(128)
instances = []
CI = ComplexIntervalField(128)
def cdiff(a, b):
    d = CI(a) - CI(b)
    return RI(d.real().abs() + d.imag().abs())
MPR = ('· rintro ⟨k, hk⟩\n  exfalso\n  have := congrArg Complex.im hk\n  simp [Complex.log_im, Complex.ofNat_arg] at this\n'
       '  have hπ : (Real.pi : ℝ) * (1 + 8 * k) = 0 := by first | linear_combination (-4) * this | linear_combination 4 * this\n'
       '  rcases mul_eq_zero.mp hπ with h | h\n  · exact Real.pi_ne_zero h\n'
       '  · have h8 : ((1 + 8 * k : ℤ) : ℝ) = 0 := by push_cast; linarith\n'
       '    have h9 : (1 + 8 * k : ℤ) = 0 := by exact_mod_cast h8\n    omega')
instances.append(kernel_instance(*['0'], proof='constructor\n· intro h\n  have := congrArg Complex.im h\n  simp at this\n' + MPR, edge=True))
instances.append(kernel_instance(*['1'], proof='constructor\n· intro h\n  have := congrArg Complex.im h\n  simp [Complex.exp_im] at this\n' + MPR, edge=False))
instances.append(kernel_instance(*['2'], proof='constructor\n· intro h\n  have := congrArg Complex.im h\n  simp [Complex.exp_im] at this\n' + MPR, edge=False))
instances.append(oracle_instance(*['(Real.log 2 / 2 + Real.pi / 4 * Complex.I)'], lhs=cdiff(exp(log(2)/2 + I*pi/4), 1 + I), rhs=RI(0), relation='==', edge=True))
print([(x.values, x.kind, x.verdict) for x in instances])
