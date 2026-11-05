# name: set-neg-not-nat-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.neg_not_nat
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['-1', '(by norm_num)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (z : ℤ) (hz : z < 0), ¬ ∃ n : ℕ, (n : ℤ) = z := by\n  intro z hz\n  rintro ⟨n, rfl⟩\n  omega\nexact @key (-1) ((by norm_num))'),
    (['-5', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (z : ℤ) (hz : z < 0), ¬ ∃ n : ℕ, (n : ℤ) = z := by\n  intro z hz\n  rintro ⟨n, rfl⟩\n  omega\nexact @key (-5) ((by norm_num))'),
    (['-100', '(by norm_num)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (z : ℤ) (hz : z < 0), ¬ ∃ n : ℕ, (n : ℤ) = z := by\n  intro z hz\n  rintro ⟨n, rfl⟩\n  omega\nexact @key (-100) ((by norm_num))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
