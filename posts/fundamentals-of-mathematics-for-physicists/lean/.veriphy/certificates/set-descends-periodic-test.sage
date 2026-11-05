# name: set-descends-periodic-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Quotients.descends_iff_periodic
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', '2 * Real.pi', 'Real.cos'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {Y : Type} (T : ℝ) (f : ℝ → Y), (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by\n  intro Y T f\n  constructor\n  · intro h x\n    exact (h x (x + T) ⟨1, by simp⟩).symm\n  · rintro h x y ⟨k, hk⟩\n    rw [show y = x + k * T by linarith, ← zsmul_eq_mul]\n    exact (h.zsmul k x).symm\nexact @key (ℝ) (2 * Real.pi) (Real.cos)'),
    (['ℝ', '0', 'id'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {Y : Type} (T : ℝ) (f : ℝ → Y), (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by\n  intro Y T f\n  constructor\n  · intro h x\n    exact (h x (x + T) ⟨1, by simp⟩).symm\n  · rintro h x y ⟨k, hk⟩\n    rw [show y = x + k * T by linarith, ← zsmul_eq_mul]\n    exact (h.zsmul k x).symm\nexact @key (ℝ) (0) (id)'),
    (['ℝ', '1', 'fun x => Real.sin (2 * Real.pi * x)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {Y : Type} (T : ℝ) (f : ℝ → Y), (∀ x y, (∃ k : ℤ, y - x = k * T) → f x = f y) ↔ Function.Periodic f T := by\n  intro Y T f\n  constructor\n  · intro h x\n    exact (h x (x + T) ⟨1, by simp⟩).symm\n  · rintro h x y ⟨k, hk⟩\n    rw [show y = x + k * T by linarith, ← zsmul_eq_mul]\n    exact (h.zsmul k x).symm\nexact @key (ℝ) (1) (fun x => Real.sin (2 * Real.pi * x))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
