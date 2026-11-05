# name: ca-num-log-mul-right-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_mul_right_half file "ca-num-log-mul-right-test.sage"
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
instances.append(kernel_instance(*['1', '1', '(by norm_num)', '(by norm_num)'], proof='simp', edge=True))
instances.append(oracle_instance(*['(2 + Complex.I)', '(1 + 3 * Complex.I)', '(by norm_num)', '(by norm_num)'], lhs=cdiff(log((2+I)*(1+3*I)), log(2+I)+log(1+3*I)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['1', '(5 - Complex.I)', '(by norm_num)', '(by norm_num)'], lhs=cdiff(log(5-I), log(1)+log(5-I)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
