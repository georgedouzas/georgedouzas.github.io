# name: set-matrix-maps-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Continuous.matrix_maps_continuous
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['3', 'KaluzaKlein.Sets.Sets.northPole'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (v : Fin n → ℝ), Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) := by\n  intro n v\n  exact (⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩)\nexact @key (3) (KaluzaKlein.Sets.Sets.northPole)'),
    (['0', '![]'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (v : Fin n → ℝ), Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) := by\n  intro n v\n  exact (⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩)\nexact @key (0) (![])'),
    (['2', '![1, 1]'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (v : Fin n → ℝ), Continuous (fun p : Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ => p.1 * p.2) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g.det) ∧ Continuous (fun g : Matrix (Fin n) (Fin n) ℝ => g *ᵥ v) := by\n  intro n v\n  exact (⟨continuous_fst.matrix_mul continuous_snd, continuous_id.matrix_det, continuous_id.matrix_mulVec continuous_const⟩)\nexact @key (2) (![1, 1])'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
