# name: set-add-descends-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.add_descends
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['5', '7', '2', '3', '8', '(by decide)', '(by decide)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]), a + b ≡ a' + b' [ZMOD n] := by\n  intro n a a' b b' ha hb\n  exact (Int.ModEq.add ha hb)\nexact @key (5) (7) (2) (3) (8) ((by decide)) ((by decide))"),
    (['0', '1', '1', '2', '2', '(by decide)', '(by decide)'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]), a + b ≡ a' + b' [ZMOD n] := by\n  intro n a a' b b' ha hb\n  exact (Int.ModEq.add ha hb)\nexact @key (0) (1) (1) (2) (2) ((by decide)) ((by decide))"),
    (['3', '-1', '2', '4', '1', '(by decide)', '(by decide)'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (n a a' b b' : ℤ) (ha : a ≡ a' [ZMOD n]) (hb : b ≡ b' [ZMOD n]), a + b ≡ a' + b' [ZMOD n] := by\n  intro n a a' b b' ha hb\n  exact (Int.ModEq.add ha hb)\nexact @key (3) (-1) (2) (4) (1) ((by decide)) ((by decide))"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
