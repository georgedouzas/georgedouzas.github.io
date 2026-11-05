# name: ca-num-arg-arctan-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arg_eq_arctan file "ca-num-arg-arctan-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', '(by norm_num)'], proof='simp', edge=True))
instances.append(oracle_instance(*['(2 + Complex.I)', '(by norm_num)'], lhs=arg(2+I), rhs=arctan(1/2), relation='==', edge=False))
instances.append(oracle_instance(*['(3 + 4 * Complex.I)', '(by norm_num)'], lhs=arg(3+4*I), rhs=arctan(4/3), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
