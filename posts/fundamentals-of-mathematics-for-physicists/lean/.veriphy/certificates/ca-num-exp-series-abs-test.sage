# name: ca-num-exp-series-abs-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_series_abs file "ca-num-exp-series-abs-test.sage"
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
n = var('n')
instances.append(oracle_instance(*['0'], lhs=RI(1), rhs=RI(exp(0)), relation='==', edge=True))
instances.append(oracle_instance(*['1'], lhs=RI(sum(1/factorial(n), n, 0, oo)), rhs=RI(exp(1)), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)'], lhs=RI(sum(sqrt(5)^n/factorial(n), n, 0, oo)), rhs=RI(exp(sqrt(5))), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
