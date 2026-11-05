# name: ca-num-inner-self-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.inner_self_eq file "ca-num-inner-self-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0'], proof='simp', edge=True))
instances.append(kernel_instance(*['1'], proof='simp', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)'], lhs=RI((QQbar(2+I)*QQbar(2+I).conjugate()).real()), rhs=RI(AA(abs(QQbar(2+I)))^2), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
