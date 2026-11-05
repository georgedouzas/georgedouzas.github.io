# name: set-x-exp-le-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Compact.x_exp_le
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by\n  intro x\n  refine ⟨?_, one_mul _⟩\n  have h := Real.add_one_le_exp (x - 1)\n  have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf\n  rw [hx]\n  have hpos := Real.exp_pos (-(x - 1))\n  have : x * Real.exp (-(x - 1)) ≤ 1 := by\n    rw [Real.exp_neg]\n    rw [mul_inv_le_iff₀ (Real.exp_pos _)]\n    linarith\n  nlinarith [Real.exp_pos (-1)]\nexact @key (1)'),
    (['0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by\n  intro x\n  refine ⟨?_, one_mul _⟩\n  have h := Real.add_one_le_exp (x - 1)\n  have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf\n  rw [hx]\n  have hpos := Real.exp_pos (-(x - 1))\n  have : x * Real.exp (-(x - 1)) ≤ 1 := by\n    rw [Real.exp_neg]\n    rw [mul_inv_le_iff₀ (Real.exp_pos _)]\n    linarith\n  nlinarith [Real.exp_pos (-1)]\nexact @key (0)'),
    (['2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (x : ℝ), x * Real.exp (-x) ≤ Real.exp (-1) ∧ 1 * Real.exp (-1) = Real.exp (-1) := by\n  intro x\n  refine ⟨?_, one_mul _⟩\n  have h := Real.add_one_le_exp (x - 1)\n  have hx : Real.exp (-x) = Real.exp (-1) * Real.exp (-(x - 1)) := by rw [← Real.exp_add]; ring_nf\n  rw [hx]\n  have hpos := Real.exp_pos (-(x - 1))\n  have : x * Real.exp (-(x - 1)) ≤ 1 := by\n    rw [Real.exp_neg]\n    rw [mul_inv_le_iff₀ (Real.exp_pos _)]\n    linarith\n  nlinarith [Real.exp_pos (-1)]\nexact @key (2)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
