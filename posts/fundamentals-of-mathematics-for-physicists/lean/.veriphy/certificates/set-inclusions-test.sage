# name: set-inclusions-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.int_real_complex_inclusions
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '0', '0', '0'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℤ) (x y : ℝ), ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) := by\n  intro a b x y\n  exact (⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩)\nexact @key (0) (0) (0) (0)'),
    (['1', '-2', '1/2', '3'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℤ) (x y : ℝ), ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) := by\n  intro a b x y\n  exact (⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩)\nexact @key (1) (-2) (1/2) (3)'),
    (['5', '5', '-1', '-1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (a b : ℤ) (x y : ℝ), ((a : ℝ) = b → a = b) ∧ ((x : ℂ) = y → x = y) := by\n  intro a b x y\n  exact (⟨fun h => by exact_mod_cast h, fun h => by exact_mod_cast h⟩)\nexact @key (5) (5) (-1) (-1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
