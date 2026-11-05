# name: ca-num-root-pow-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.root_of_unity_pow file "ca-num-root-pow-test.sage"
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
instances.append(kernel_instance(*['1', '0', 'one_ne_zero'], proof='simp', edge=True))
instances.append(oracle_instance(*['6', '1', '(by norm_num)'], lhs=cdiff(exp(2*pi*I*1/6)^6, 1), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['4', '3', '(by norm_num)'], lhs=cdiff(exp(2*pi*I*3/4)^4, 1), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
