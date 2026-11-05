# name: ca-num-fout-imag-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.Branches.fOut_imaginary file "ca-num-fout-imag-test.sage"
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
instances.append(oracle_instance(*['1', 'one_pos'], lhs=cdiff(I*(1-1/I^2)^(1/2), sqrt(2)*I), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['2', 'two_pos'], lhs=cdiff(2*I*(1-1/(2*I)^2)^(1/2), sqrt(5)*I), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(1 / 2)', '(by norm_num)'], lhs=cdiff(I/2*(1-1/(I/2)^2)^(1/2), sqrt(5/4)*I), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
