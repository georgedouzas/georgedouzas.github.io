# name: ca-ex-vertical-bound-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Exercises.vertical_side_bound file "ca-ex-vertical-bound-test.sage"
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
instances.append(kernel_instance(*['0', '0', '0', 'le_rfl'], proof='simp', edge=True))
instances.append(oracle_instance(*['2', '(1 / 2)', '1', '(by norm_num)'], lhs=RI(RI(abs(CI(exp(-(2+I/2)^2/2))))), rhs=RI(RI(exp(-2+1/2))), relation='<=', edge=False))
instances.append(oracle_instance(*['3', '(-1 / 2)', '2', '(by norm_num)'], lhs=RI(RI(abs(CI(exp(-(3-I/2)^2/2))))), rhs=RI(RI(exp(-9/2+2))), relation='<=', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
