# name: ca-num-cube-root-neg-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.neg_eight_cpow_third file "ca-num-cube-root-neg-test.sage"
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
instances.append(oracle_instance(*['1', 'one_pos'], lhs=cdiff((-1)^(1/3), exp(pi/3*I)), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['2', 'two_pos'], lhs=cdiff((-8)^(1/3), 1+sqrt(3)*I), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['3', '(by norm_num)'], lhs=cdiff((-27)^(1/3), 3*exp(pi/3*I)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
