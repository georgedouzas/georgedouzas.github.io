# name: ca-num-circle-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.circle_mul_inv file "ca-num-circle-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', '1', 'norm_one', 'norm_one'], proof='constructor <;> simp', edge=True))
instances.append(kernel_instance(*['Complex.I', '(-1)', 'Complex.norm_I', '(by simp)'], proof='constructor <;> simp', edge=False))
instances.append(kernel_instance(*['(-Complex.I)', 'Complex.I', '(by simp)', 'Complex.norm_I'], proof='constructor <;> simp', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
