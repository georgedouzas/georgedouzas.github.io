# name: ca-num-arg-inv-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arg_inv_mod file "ca-num-arg-inv-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1'], proof='simp', edge=True))
instances.append(kernel_instance(*['Complex.I'], proof='rw [Complex.inv_I, Complex.arg_neg_I, Complex.arg_I]\nsimp', edge=False))
instances.append(kernel_instance(*['(-1)'], proof='rw [inv_neg, inv_one, Complex.arg_neg_one]\nexact Real.Angle.neg_coe_pi.symm', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
