# name: ca-num-sq-norm-add-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_add file "ca-num-sq-norm-add-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['1', 'Complex.I'], lhs=RI(AA(abs(QQbar(1+I)))^2), rhs=RI(AA(abs(QQbar(1)))^2 + AA(abs(QQbar(I)))^2 + 2*(QQbar(1)*QQbar(I).conjugate()).real()), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '(1 - 3 * Complex.I)'], lhs=RI(AA(abs(QQbar(3-2*I)))^2), rhs=RI(AA(abs(QQbar(2+I)))^2 + AA(abs(QQbar(1-3*I)))^2 + 2*(QQbar(2+I)*QQbar(1-3*I).conjugate()).real()), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
