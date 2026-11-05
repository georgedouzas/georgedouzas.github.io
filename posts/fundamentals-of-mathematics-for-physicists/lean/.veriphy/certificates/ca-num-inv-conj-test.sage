# name: ca-num-inv-conj-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks or Sage evaluates exactly
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.inv_eq_conj_div file "ca-num-inv-conj-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', 'one_ne_zero'], proof='simp', edge=True))
instances.append(oracle_instance(*['Complex.I', 'Complex.I_ne_zero'], lhs=QQbar(I)*(QQbar(I).conjugate()/(QQbar(I)*QQbar(I).conjugate())), rhs=QQbar(1), relation='==', edge=False))
instances.append(oracle_instance(*['(2 + Complex.I)', '(by intro h; have := congrArg Complex.re h; norm_num at this)'], lhs=QQbar(2+I)*(QQbar(2+I).conjugate()/(QQbar(2+I)*QQbar(2+I).conjugate())), rhs=QQbar(1), relation='==', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
