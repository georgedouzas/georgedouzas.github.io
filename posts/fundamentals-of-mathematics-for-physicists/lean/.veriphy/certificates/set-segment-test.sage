# name: set-segment-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Connected.segment_joined
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['0', '![]', '![]'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p q : Fin n → ℝ), JoinedIn univ p q := by\n  intro n p q\n  exact (JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _))\nexact @key (0) (![]) (![])'),
    (['2', '![0, 0]', '![1, 1]'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p q : Fin n → ℝ), JoinedIn univ p q := by\n  intro n p q\n  exact (JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _))\nexact @key (2) (![0, 0]) (![1, 1])'),
    (['3', '![1, 2, 3]', '![0, 0, 0]'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (p q : Fin n → ℝ), JoinedIn univ p q := by\n  intro n p q\n  exact (JoinedIn.ofLine (f := fun t : ℝ => (1 - t) • p + t • q) (by fun_prop) (by simp) (by simp) (subset_univ _))\nexact @key (3) (![1, 2, 3]) (![0, 0, 0])'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
