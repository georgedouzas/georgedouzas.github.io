# name: ca-num-norm-mul-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.norm_mul_eq file "ca-num-norm-mul-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '(2 + Complex.I)'], proof='simp', edge=True))
instances.append(oracle_instance(*['(3 + 4 * Complex.I)', '(1 - Complex.I)'], lhs=RI(abs(QQbar((3+4*I)*(1-I)))), rhs=RI(abs(QQbar(3+4*I))*abs(QQbar(1-I))), relation='==', edge=False))
instances.append(oracle_instance(*['Complex.I', '2'], lhs=RI(abs(QQbar(2*I))), rhs=RI(abs(QQbar(I))*abs(QQbar(2))), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
