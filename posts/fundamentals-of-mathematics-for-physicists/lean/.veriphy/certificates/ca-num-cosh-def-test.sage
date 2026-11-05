# name: ca-num-cosh-def-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.cosh_def' file "ca-num-cosh-def-test.sage"
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
instances.append(kernel_instance(*['0'], proof='simp', edge=True))
instances.append(oracle_instance(*['1'], lhs=cdiff(cosh(1), (exp((1))+exp(-(1)))/2), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['Complex.I'], lhs=cdiff(cosh(I), (exp((I))+exp(-(I)))/2), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)'], lhs=cdiff(cosh(2+I), (exp((2+I))+exp(-(2+I)))/2), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
