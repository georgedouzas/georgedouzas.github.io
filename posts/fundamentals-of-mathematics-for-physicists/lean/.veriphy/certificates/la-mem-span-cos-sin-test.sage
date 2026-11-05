# name: la-mem-span-cos-sin-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Spans.mem_span_cos_sin file "la-mem-span-cos-sin-test.sage"
# description: f ∈ span(cos, sin) ↔ f = a cos + b sin at f = 0, cos and 2 cos - sin, with the coefficients exhibited.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True, 'refine ⟨fun _ => ⟨0, 0, ?_⟩, fun _ => ?_⟩\n· funext φ; simp <;> ring\n· rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]\n  exact ⟨0, 0, by funext φ; simp <;> ring⟩'),
    (['Real.cos'], False, 'refine ⟨fun _ => ⟨1, 0, ?_⟩, fun _ => ?_⟩\n· funext φ; simp <;> ring\n· rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]\n  exact ⟨1, 0, by funext φ; simp <;> ring⟩'),
    (['fun φ => 2 * Real.cos φ - Real.sin φ'], False, 'refine ⟨fun _ => ⟨2, -1, ?_⟩, fun _ => ?_⟩\n· funext φ; simp <;> ring\n· rw [Matrix.range_cons_cons_empty, Submodule.mem_span_pair]\n  exact ⟨2, -1, by funext φ; simp <;> ring⟩'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
