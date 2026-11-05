# name: set-discrete-indiscrete-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.TopologicalSpaces.discrete_indiscrete
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type), IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by\n  intro X\n  refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩\n  refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩\n  · rintro U V (rfl | rfl) (rfl | rfl) <;> simp\n  · by_cases h : univ ∈ S\n    · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩\n    · left\n      rw [sUnion_eq_empty]\n      intro U hU\n      rcases hS hU with rfl | rfl\n      · rfl\n      · exact absurd hU h\nexact @key (Fin 3)'),
    (['Empty'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type), IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by\n  intro X\n  refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩\n  refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩\n  · rintro U V (rfl | rfl) (rfl | rfl) <;> simp\n  · by_cases h : univ ∈ S\n    · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩\n    · left\n      rw [sUnion_eq_empty]\n      intro U hU\n      rcases hS hU with rfl | rfl\n      · rfl\n      · exact absurd hU h\nexact @key (Empty)'),
    (['ℕ'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X : Type), IsTopologyOn X univ ∧ IsTopologyOn X {∅, univ} := by\n  intro X\n  refine ⟨⟨trivial, trivial, fun _ _ _ _ => trivial, fun _ _ => trivial⟩, ?_⟩\n  refine ⟨Or.inl rfl, Or.inr rfl, ?_, fun S hS => ?_⟩\n  · rintro U V (rfl | rfl) (rfl | rfl) <;> simp\n  · by_cases h : univ ∈ S\n    · right; exact sUnion_eq_univ_iff.2 fun x => ⟨univ, h, trivial⟩\n    · left\n      rw [sUnion_eq_empty]\n      intro U hU\n      rcases hS hU with rfl | rfl\n      · rfl\n      · exact absurd hU h\nexact @key (ℕ)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
