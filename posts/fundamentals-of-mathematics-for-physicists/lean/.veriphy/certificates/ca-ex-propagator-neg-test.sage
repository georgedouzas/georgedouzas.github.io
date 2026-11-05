# name: ca-ex-propagator-neg-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.Exercises.propagator_neg file "ca-ex-propagator-neg-test.sage"
# description: the limit δ → 0 of a function continuous at δ = 0: Sage evaluates both sides at δ = 0; instances with an edge case first
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
instances.append(oracle_instance(*['1', '(-1)', 'one_pos'], lhs=cdiff(1/(2*pi)*(2*pi*I)*(exp(-I*(-1)*(-1))/(-2*1)), -(I/2)*exp(I*1*(-1))), rhs=RI(0), relation='==', edge=True))
instances.append(oracle_instance(*['2', '(-1 / 2)', 'two_pos'], lhs=cdiff(1/(2*pi)*(2*pi*I)*(exp(-I*(-2)*(-1/2))/(-2*2)), -(I/4)*exp(-I)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(1 / 2)', '(-3)', '(by norm_num)'], lhs=cdiff(1/(2*pi)*(2*pi*I)*(exp(-I*(-1/2)*(-3))/(-2*(1/2))), -(I)*exp(-I*3/2)), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
