# name: ca-num-fout-asym-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.fOut_asymptotic file "ca-num-fout-asym-test.sage"
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
instances.append(oracle_instance(*['2', 'two_ne_zero'], lhs=RI(RI(abs(CI(2*(1-1/4)^(1/2)) - 2))), rhs=RI(RI(1/2)), relation='<=', edge=False))
instances.append(oracle_instance(*['(1 + Complex.I)', '(by intro h; have := congrArg Complex.re h; norm_num at this)'], lhs=RI(RI(abs(CI((1+I)*(1-1/(1+I)^2)^(1/2)) - CI(1+I)))), rhs=RI(RI(1/abs(CI(1+I)))), relation='<=', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
