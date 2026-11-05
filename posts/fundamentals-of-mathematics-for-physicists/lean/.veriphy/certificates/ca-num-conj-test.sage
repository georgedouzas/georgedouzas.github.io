# name: ca-num-conj-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.conj_add_mul file "ca-num-conj-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='constructor <;> simp', edge=True))
instances.append(kernel_instance(*['1', 'Complex.I'], proof='constructor <;> apply Complex.ext <;> simp', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '(3 - Complex.I)'], lhs=(QQbar(2+I)*QQbar(3-I)).conjugate(), rhs=QQbar(2+I).conjugate()*QQbar(3-I).conjugate(), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
