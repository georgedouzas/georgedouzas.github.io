# name: set-eps-delta-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Continuous.continuous_iff_eps_delta
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'Real.sin'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : MetricSpace X] [inst2 : MetricSpace Y] (f : X → Y), Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε := by\n  intro X Y inst1 inst2 f\n  exact (Metric.continuous_iff)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (Real.sin)'),
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'id'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : MetricSpace X] [inst2 : MetricSpace Y] (f : X → Y), Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε := by\n  intro X Y inst1 inst2 f\n  exact (Metric.continuous_iff)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (id)'),
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'fun x => x ^ 2'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : MetricSpace X] [inst2 : MetricSpace Y] (f : X → Y), Continuous f ↔ ∀ p, ∀ ε > 0, ∃ δ > 0, ∀ x, dist x p < δ → dist (f x) (f p) < ε := by\n  intro X Y inst1 inst2 f\n  exact (Metric.continuous_iff)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (fun x => x ^ 2)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
