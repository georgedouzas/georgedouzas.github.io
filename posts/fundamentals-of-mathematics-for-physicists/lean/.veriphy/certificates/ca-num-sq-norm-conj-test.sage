# name: ca-num-sq-norm-conj-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.sq_norm_eq_mul_conj file "ca-num-sq-norm-conj-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0'], proof='simp', edge=True))
instances.append(kernel_instance(*['1'], proof='simp', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)'], lhs=RI(AA(abs(QQbar(2+I)))^2), rhs=RI(QQbar(2+I)*QQbar(2+I).conjugate()), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
