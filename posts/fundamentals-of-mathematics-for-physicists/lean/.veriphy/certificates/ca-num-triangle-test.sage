# name: ca-num-triangle-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.triangle file "ca-num-triangle-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='simp', edge=True))
instances.append(oracle_instance(*['1', 'Complex.I'], lhs=RI(abs(QQbar(1+I))), rhs=RI(abs(QQbar(1))+abs(QQbar(I))), relation='<=', edge=False))
instances.append(oracle_instance(*['(3 + 4 * Complex.I)', '(-1 + 2 * Complex.I)'], lhs=RI(abs(QQbar(2+6*I))), rhs=RI(abs(QQbar(3+4*I))+abs(QQbar(-1+2*I))), relation='<=', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
