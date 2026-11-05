# name: set-remainders-distinct-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.remainders_distinct
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['5', '2', '2', '(by norm_num)', '(by norm_num)', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]), r = s := by\n  intro n r s hr hs h\n  rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h\n  exact_mod_cast h\nexact @key (5) (2) (2) ((by norm_num)) ((by norm_num)) ((by decide))'),
    (['1', '0', '0', '(by norm_num)', '(by norm_num)', '(by decide)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]), r = s := by\n  intro n r s hr hs h\n  rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h\n  exact_mod_cast h\nexact @key (1) (0) (0) ((by norm_num)) ((by norm_num)) ((by decide))'),
    (['12', '7', '7', '(by norm_num)', '(by norm_num)', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n r s : ℕ) (hr : r < n) (hs : s < n) (h : (r : ℤ) ≡ s [ZMOD n]), r = s := by\n  intro n r s hr hs h\n  rw [Int.ModEq, Int.emod_eq_of_lt (by omega) (by omega), Int.emod_eq_of_lt (by omega) (by omega)] at h\n  exact_mod_cast h\nexact @key (12) (7) (7) ((by norm_num)) ((by norm_num)) ((by decide))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
