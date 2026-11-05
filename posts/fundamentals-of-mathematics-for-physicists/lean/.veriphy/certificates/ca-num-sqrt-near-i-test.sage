# name: ca-num-sqrt-near-i-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.sqrt_near_I file "ca-num-sqrt-near-i-test.sage"
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
instances.append(kernel_instance(*['Complex.I', '0', '(by simp [Complex.I_sq])', '(by simp)'], proof='simp', edge=True))
instances.append(oracle_instance(*['(1 + Real.sqrt 2 * Complex.I)', '(2 * Real.sqrt 2)', '(by have h : ((Real.sqrt 2 : ℝ) : ℂ) ^ 2 = 2 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2); push_cast; linear_combination h + 2 * Complex.I_sq)', '(by simp)'], lhs=RI(RI(abs(CI(1+sqrt(2)*I-I)))), rhs=RI(RI(abs(2*sqrt(2)))), relation='<=', edge=False))
instances.append(oracle_instance(*['(2 + Real.sqrt 5 * Complex.I)', '(4 * Real.sqrt 5)', '(by have h : ((Real.sqrt 5 : ℝ) : ℂ) ^ 2 = 5 := by exact_mod_cast Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 5); push_cast; linear_combination h + 5 * Complex.I_sq)', '(by simp)'], lhs=RI(RI(abs(CI(2+sqrt(5)*I-I)))), rhs=RI(RI(abs(4*sqrt(5)))), relation='<=', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
