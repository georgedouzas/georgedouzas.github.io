# name: set-parametrisations-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Continuous.parametrisations_continuous
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r : ℝ), Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧ Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by\n  intro r\n  have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by\n    refine continuous_pi fun i => ?_\n    fin_cases i <;> simp [sph] <;> fun_prop\n  exact ⟨by fun_prop, hs.const_smul r⟩\nexact @key (1)'),
    (['0'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r : ℝ), Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧ Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by\n  intro r\n  have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by\n    refine continuous_pi fun i => ?_\n    fin_cases i <;> simp [sph] <;> fun_prop\n  exact ⟨by fun_prop, hs.const_smul r⟩\nexact @key (0)'),
    (['2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (r : ℝ), Continuous (fun φ : ℝ => (r * Real.cos φ, r * Real.sin φ)) ∧ Continuous (fun q : ℝ × ℝ => r • sph q.1 q.2) := by\n  intro r\n  have hs : Continuous (fun q : ℝ × ℝ => sph q.1 q.2) := by\n    refine continuous_pi fun i => ?_\n    fin_cases i <;> simp [sph] <;> fun_prop\n  exact ⟨by fun_prop, hs.const_smul r⟩\nexact @key (2)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
