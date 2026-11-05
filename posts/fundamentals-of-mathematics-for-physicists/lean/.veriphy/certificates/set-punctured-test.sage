# name: set-punctured-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.OpenSets.punctured_open
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['2', '0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p : Fin n → ℝ), IsOpen ({p}ᶜ : Set (Fin n → ℝ)) := by\n  intro n p\n  exact (isOpen_compl_singleton)\nexact @key (2) (0)'),
    (['0', '![]'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p : Fin n → ℝ), IsOpen ({p}ᶜ : Set (Fin n → ℝ)) := by\n  intro n p\n  exact (isOpen_compl_singleton)\nexact @key (0) (![])'),
    (['3', '![1, 2, 3]'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p : Fin n → ℝ), IsOpen ({p}ᶜ : Set (Fin n → ℝ)) := by\n  intro n p\n  exact (isOpen_compl_singleton)\nexact @key (3) (![1, 2, 3])'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
