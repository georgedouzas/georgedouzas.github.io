# name: set-heine-borel-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Compact.heine_borel
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'inferInstance', 'inferInstance', 'Set.Icc 0 1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E), IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K := by\n  intro E inst1 inst2 K\n  exact (Metric.isCompact_iff_isClosed_bounded)\nexact @key (ℝ) (inferInstance) (inferInstance) (Set.Icc 0 1)'),
    (['ℝ', 'inferInstance', 'inferInstance', '∅'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E), IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K := by\n  intro E inst1 inst2 K\n  exact (Metric.isCompact_iff_isClosed_bounded)\nexact @key (ℝ) (inferInstance) (inferInstance) (∅)'),
    (['ℝ × ℝ', 'inferInstance', 'inferInstance', 'KaluzaKlein.Sets.Sets.circle'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (E : Type) [inst1 : MetricSpace E] [inst2 : ProperSpace E] (K : Set E), IsCompact K ↔ IsClosed K ∧ Bornology.IsBounded K := by\n  intro E inst1 inst2 K\n  exact (Metric.isCompact_iff_isClosed_bounded)\nexact @key (ℝ × ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Sets.circle)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
