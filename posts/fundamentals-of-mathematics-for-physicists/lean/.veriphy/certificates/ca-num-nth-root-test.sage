# name: ca-num-nth-root-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.nth_root_values file "ca-num-nth-root-test.sage"
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
instances.append(kernel_instance(*['1', 'one_ne_zero', '1', 'one_ne_zero', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['(-8)', '(by norm_num)', '3', '(by norm_num)', '1'], lhs=cdiff(exp((log(-8)+2*pi*I)/3)^3, -8), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['Complex.I', 'Complex.I_ne_zero', '2', 'two_ne_zero', '3'], lhs=cdiff(exp((log(I)+6*pi*I)/2)^2, I), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
