# name: ca-num-arg-range-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arg_mem_Ioc' file "ca-num-arg-range-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0'], proof='simp only [Complex.arg_zero]\nconstructor <;> linarith [Real.pi_pos]', edge=True))
instances.append(kernel_instance(*['Complex.I'], proof='simp\nconstructor <;> linarith [Real.pi_pos]', edge=False))
instances.append(kernel_instance(*['(-1)'], proof='simp [Real.pi_pos]', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
