# name: ca-num-exp-onto-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexExponential.exp_surjective' file "ca-num-exp-onto-test.sage"
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
instances.append(kernel_instance(*['1', 'one_ne_zero', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['Complex.I', 'Complex.I_ne_zero', '1'], lhs=cdiff(exp(log(1)+pi/2*I+2*pi*I), I), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(-2)', '(by norm_num)', '(-1)'], lhs=cdiff(exp(log(2)+pi*I-2*pi*I), -2), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
