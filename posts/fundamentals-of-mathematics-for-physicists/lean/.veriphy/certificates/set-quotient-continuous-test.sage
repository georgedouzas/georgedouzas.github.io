# name: set-quotient-continuous-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.QuotientTopology.quotient_continuous_iff
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Unit', 'Unit', 'inferInstance', 'inferInstance', '⊥', 'fun _ => ()'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (s : Setoid X) (g : Quotient s → Y), Continuous g ↔ Continuous (g ∘ Quotient.mk s) := by\n  intro X Y inst1 inst2 s g\n  exact (isQuotientMap_quotient_mk'.continuous_iff)\nexact @key (Unit) (Unit) (inferInstance) (inferInstance) (⊥) (fun _ => ())"),
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', 'KaluzaKlein.Sets.Quotients.angleSetoid 1', 'fun _ => 0'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (s : Setoid X) (g : Quotient s → Y), Continuous g ↔ Continuous (g ∘ Quotient.mk s) := by\n  intro X Y inst1 inst2 s g\n  exact (isQuotientMap_quotient_mk'.continuous_iff)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (KaluzaKlein.Sets.Quotients.angleSetoid 1) (fun _ => 0)"),
    (['ℝ', 'ℝ', 'inferInstance', 'inferInstance', '⊤', 'fun _ => 1'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : TopologicalSpace X] [inst2 : TopologicalSpace Y] (s : Setoid X) (g : Quotient s → Y), Continuous g ↔ Continuous (g ∘ Quotient.mk s) := by\n  intro X Y inst1 inst2 s g\n  exact (isQuotientMap_quotient_mk'.continuous_iff)\nexact @key (ℝ) (ℝ) (inferInstance) (inferInstance) (⊤) (fun _ => 1)"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
