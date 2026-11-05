# name: ca-num-neg-one-root-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.neg_one_cpow_third file "ca-num-neg-one-root-test.sage"
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
instances.append(oracle_instance(*['1', 'one_ne_zero'], lhs=cdiff((-1)^(1/1), exp(pi/1*I)), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['3', '(by norm_num)'], lhs=cdiff((-1)^(1/3), 1/2+sqrt(3)/2*I), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['2', 'two_ne_zero'], lhs=cdiff((-1)^(1/2), I), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
