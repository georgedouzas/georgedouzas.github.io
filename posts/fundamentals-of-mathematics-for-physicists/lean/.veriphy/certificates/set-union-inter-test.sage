# name: set-union-inter-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.OpenSets.union_inter_open
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1', '{Metric.ball 0 1, Metric.ball 0 2}', '(by rintro U (rfl | rfl) <;> exact Metric.isOpen_ball)', '{Metric.ball 0 1}', '(by simp [Metric.isOpen_ball])'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U) (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U), IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) := by\n  intro n S hS F hF\n  exact (⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩)\nexact @key (1) ({Metric.ball 0 1, Metric.ball 0 2}) ((by rintro U (rfl | rfl) <;> exact Metric.isOpen_ball)) ({Metric.ball 0 1}) ((by simp [Metric.isOpen_ball]))'),
    (['0', '∅', '(by simp)', '∅', '(by simp)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U) (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U), IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) := by\n  intro n S hS F hF\n  exact (⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩)\nexact @key (0) (∅) ((by simp)) (∅) ((by simp))'),
    (['2', '{Set.univ}', '(by simp)', '{Metric.ball 0 3}', '(by simp [Metric.isOpen_ball])'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (n : ℕ) (S : Set (Set (EuclideanSpace ℝ (Fin n)))) (hS : ∀ U ∈ S, IsOpen U) (F : Finset (Set (EuclideanSpace ℝ (Fin n)))) (hF : ∀ U ∈ F, IsOpen U), IsOpen (⋃₀ S) ∧ IsOpen (⋂₀ (F : Set (Set (EuclideanSpace ℝ (Fin n))))) := by\n  intro n S hS F hF\n  exact (⟨isOpen_sUnion hS, (F.finite_toSet).isOpen_sInter hF⟩)\nexact @key (2) ({Set.univ}) ((by simp)) ({Metric.ball 0 3}) ((by simp [Metric.isOpen_ball]))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
