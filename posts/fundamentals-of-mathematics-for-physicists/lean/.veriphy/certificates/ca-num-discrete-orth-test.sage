# name: ca-num-discrete-orth-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.NumbersWorked.discrete_orthogonality file "ca-num-discrete-orth-test.sage"
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
instances.append(oracle_instance(*['2', '0', 'le_rfl'], lhs=cdiff(sum(exp(2*pi*I/2)^(k*0) for k in range(2)), 2), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['6', '1', '(by norm_num)'], lhs=cdiff(sum(exp(2*pi*I/6)^(k*1) for k in range(6)), 0), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['6', '6', '(by norm_num)'], lhs=cdiff(sum(exp(2*pi*I/6)^(k*6) for k in range(6)), 6), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['4', '2', '(by norm_num)'], lhs=cdiff(sum(exp(2*pi*I/4)^(k*2) for k in range(4)), 0), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
