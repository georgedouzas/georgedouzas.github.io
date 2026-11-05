# name: ca-num-log-mul-iff-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_iff file "ca-num-log-mul-iff-test.sage"
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
instances.append(kernel_instance(*['1', '1', 'one_ne_zero', 'one_ne_zero'], proof='simp [Real.pi_pos, Real.pi_pos.le]', edge=True))
instances.append(kernel_instance(*['(-1)', '(-1)', '(by norm_num)', '(by norm_num)'], proof='rw [show ((-1 : ℂ) * -1) = 1 by norm_num, Complex.log_one, Complex.log_neg_one, Complex.arg_neg_one]\nconstructor\n· intro h\n  have := congrArg Complex.im h\n  simp at this\n  linarith [Real.pi_pos]\n· intro h\n  simp at h\n  linarith [h.2, Real.pi_pos]', edge=False))
instances.append(kernel_instance(*['1', '(-1)', 'one_ne_zero', '(by norm_num)'], proof='simp [Complex.arg_neg_one, Real.pi_pos]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
