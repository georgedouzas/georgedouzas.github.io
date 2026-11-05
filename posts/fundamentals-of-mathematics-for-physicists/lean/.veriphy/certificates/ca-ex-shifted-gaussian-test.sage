# name: ca-ex-shifted-gaussian-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.Exercises.shifted_gaussian file "ca-ex-shifted-gaussian-test.sage"
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
x = var('x')
instances.append(oracle_instance(*['0'], lhs=cdiff(integrate(exp(-x^2/2), x, -oo, oo), sqrt(2*pi)), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['1'], lhs=cdiff(integrate(exp(-(x-I)^2/2), x, -oo, oo), sqrt(2*pi)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(-2)'], lhs=cdiff(integrate(exp(-(x+2*I)^2/2), x, -oo, oo), sqrt(2*pi)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
