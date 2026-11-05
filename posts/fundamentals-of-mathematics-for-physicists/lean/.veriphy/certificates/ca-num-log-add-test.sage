# name: ca-num-log-add-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexLogarithm.log_add_is_log file "ca-num-log-add-test.sage"
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
instances.append(kernel_instance(*['1', '1', 'one_ne_zero', 'one_ne_zero'], proof='simp', edge=True))
instances.append(oracle_instance(*['Complex.I', '(-1)', 'Complex.I_ne_zero', '(by norm_num)'], lhs=cdiff(exp(log(I)+log(-1)), I*(-1)), rhs=RI(0), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '3', '(by intro h; have := congrArg Complex.re h; norm_num at this)', '(by norm_num)'], lhs=cdiff(exp(log(2+I)+log(3)), (2+I)*3), rhs=RI(0), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
