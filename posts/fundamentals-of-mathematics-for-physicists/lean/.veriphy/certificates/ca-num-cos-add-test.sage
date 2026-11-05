# name: ca-num-cos-add-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.cos_add' file "ca-num-cos-add-test.sage"
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
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['1', 'Complex.I'], lhs=cdiff(cos(1+I), cos(1)*cos(I)-sin(1)*sin(I)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '(1 / 2)'], lhs=cdiff(cos(5/2+I), cos(2+I)*cos(1/2)-sin(2+I)*sin(1/2)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
