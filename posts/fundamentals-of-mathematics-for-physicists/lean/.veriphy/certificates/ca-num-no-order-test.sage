# name: ca-num-no-order-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.ComplexNumbers.no_positive_cone file "ca-num-no-order-test.sage"
# description: for every candidate set of positive numbers the axioms of an order fail; instances are sets where a specific axiom fails; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['∅'], proof='rintro ⟨-, -, htri, -⟩\nsimpa using htri 1 one_ne_zero', edge=True))
instances.append(kernel_instance(*['Set.univ'], proof='rintro ⟨-, -, -, h0⟩\nexact h0 (Set.mem_univ _)', edge=False))
instances.append(kernel_instance(*['{z : ℂ | 0 < z.re}'], proof='rintro ⟨-, -, htri, -⟩\nrcases htri Complex.I Complex.I_ne_zero with h | h <;> simp at h', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
