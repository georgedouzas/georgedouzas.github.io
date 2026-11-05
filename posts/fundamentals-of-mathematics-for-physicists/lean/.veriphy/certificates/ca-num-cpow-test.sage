# name: ca-num-cpow-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.cpow_eq file "ca-num-cpow-test.sage"
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
instances.append(oracle_instance(*['Complex.I', '(1 / 2)', 'Complex.I_ne_zero'], lhs=cdiff(I^(1/2), exp(log(I)/2)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['2', 'Complex.I', 'two_ne_zero'], lhs=cdiff(2^I, exp(I*log(2))), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
