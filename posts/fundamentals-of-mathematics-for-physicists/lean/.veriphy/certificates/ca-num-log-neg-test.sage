# name: ca-num-log-neg-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_neg_pos file "ca-num-log-neg-test.sage"
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
instances.append(kernel_instance(*['1', 'one_pos'], proof='simp [Complex.log_neg_one]', edge=True))
instances.append(oracle_instance(*['(Real.exp 1)', '(Real.exp_pos 1)'], lhs=cdiff(log(-e), 1+pi*I), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['2', 'two_pos'], lhs=cdiff(log(-2), log(2)+pi*I), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
