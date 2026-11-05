# name: ca-num-sqrt-neg-sq-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.sqrt_neg_sq file "ca-num-sqrt-neg-sq-test.sage"
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
instances.append(kernel_instance(*['1', 'one_pos'], proof='rw [show (-((1 : ℝ) : ℂ) ^ 2) = -1 by push_cast; ring]\nrw [Complex.cpow_def_of_ne_zero (by norm_num), Complex.log_neg_one]\npush_cast\nrw [show (Real.pi : ℂ) * Complex.I * (1 / 2) = Real.pi / 2 * Complex.I by ring, Complex.exp_pi_div_two_mul_I]\nsimp', edge=True))
instances.append(oracle_instance(*['2', 'two_pos'], lhs=cdiff((-4)^(1/2), 2*I), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(1 / 2)', '(by norm_num)'], lhs=cdiff((-1/4)^(1/2), I/2), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
