# name: ca-num-parallelogram-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.parallelogram file "ca-num-parallelogram-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['1', 'Complex.I'], lhs=RI(AA(abs(QQbar(1+I)))^2 + AA(abs(QQbar(1-I)))^2), rhs=RI(2*AA(abs(QQbar(1)))^2 + 2*AA(abs(QQbar(I)))^2), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '(1 - 3 * Complex.I)'], lhs=RI(AA(abs(QQbar(3-2*I)))^2 + AA(abs(QQbar(1+4*I)))^2), rhs=RI(2*AA(abs(QQbar(2+I)))^2 + 2*AA(abs(QQbar(1-3*I)))^2), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
