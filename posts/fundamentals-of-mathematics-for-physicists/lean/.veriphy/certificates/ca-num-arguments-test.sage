# name: ca-num-arguments-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arguments_iff file "ca-num-arguments-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['1', 'one_ne_zero', '0'], proof='exact ⟨fun _ => ⟨0, by simp⟩, fun _ => by simp⟩', edge=True))
instances.append(kernel_instance(*['1', 'one_ne_zero', '(2 * Real.pi)'], proof='exact ⟨fun _ => ⟨1, by simp⟩, fun _ => by push_cast; simp⟩', edge=False))
instances.append(kernel_instance(*['Complex.I', 'Complex.I_ne_zero', '(Real.pi / 2)'], proof='exact ⟨fun _ => ⟨0, by simp⟩, fun _ => by push_cast; simp⟩', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
