# name: set-modeq-equivalence-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.modEq_equivalence
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℤ), Equivalence (fun a b : ℤ => n ∣ a - b) := by\n  intro n\n  exact (⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,\n    fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩)\nexact @key (3)'),
    (['0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℤ), Equivalence (fun a b : ℤ => n ∣ a - b) := by\n  intro n\n  exact (⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,\n    fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩)\nexact @key (0)'),
    (['2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℤ), Equivalence (fun a b : ℤ => n ∣ a - b) := by\n  intro n\n  exact (⟨fun a => by simp, fun h => by rw [← neg_sub]; exact (dvd_neg).2 h,\n    fun h1 h2 => by have := dvd_add h1 h2; rwa [sub_add_sub_cancel] at this⟩)\nexact @key (2)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
