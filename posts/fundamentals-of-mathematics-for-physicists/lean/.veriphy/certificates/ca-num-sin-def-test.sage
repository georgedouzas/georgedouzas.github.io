# name: ca-num-sin-def-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.sin_def' file "ca-num-sin-def-test.sage"
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
instances.append(oracle_instance(*['1'], lhs=cdiff(sin(1), (exp((1)*I)-exp(-(1)*I))/(2*I)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['Complex.I'], lhs=cdiff(sin(I), (exp((I)*I)-exp(-(I)*I))/(2*I)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)'], lhs=cdiff(sin(2+I), (exp((2+I)*I)-exp(-(2+I)*I))/(2*I)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
