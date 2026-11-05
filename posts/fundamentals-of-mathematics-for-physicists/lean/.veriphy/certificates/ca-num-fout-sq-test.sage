# name: ca-num-fout-sq-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.fOut_sq file "ca-num-fout-sq-test.sage"
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
instances.append(kernel_instance(*['1', 'one_ne_zero'], proof='simp [fOut]', edge=True))
instances.append(oracle_instance(*['2', 'two_ne_zero'], lhs=cdiff((2*(1-1/4)^(1/2))^2, 2^2-1), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['Complex.I', 'Complex.I_ne_zero'], lhs=cdiff((I*(1-1/I^2)^(1/2))^2, I^2-1), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
