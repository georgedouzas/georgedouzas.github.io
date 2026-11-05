# name: ca-num-re-le-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.re_mul_conj_le file "ca-num-re-le-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['1', 'Complex.I'], lhs=RI((QQbar(1)*QQbar(I).conjugate()).real()), rhs=RI(abs(QQbar(1))*abs(QQbar(I))), relation='<=', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '(1 - 3 * Complex.I)'], lhs=RI((QQbar(2+I)*QQbar(1-3*I).conjugate()).real()), rhs=RI(abs(QQbar(2+I))*abs(QQbar(1-3*I))), relation='<=', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
