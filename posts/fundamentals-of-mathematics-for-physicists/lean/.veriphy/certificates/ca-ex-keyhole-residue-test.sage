# name: ca-ex-keyhole-residue-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.Exercises.keyhole_residue file "ca-ex-keyhole-residue-test.sage"
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
instances.append(oracle_instance(*['0'], lhs=cdiff(exp(pi*I*(0-1)), -exp(0)), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['(1 / 3)'], lhs=cdiff(exp(pi*I*(1/3-1)), -exp(pi*I/3)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(1 / 2)'], lhs=cdiff(exp(pi*I*(1/2-1)), -exp(pi*I/2)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
