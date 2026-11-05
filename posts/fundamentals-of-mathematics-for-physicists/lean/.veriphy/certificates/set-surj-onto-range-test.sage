# name: set-surj-onto-range-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Surjective.surjective_onto_range
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', 'Fin 2', 'id'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Surjective (Set.rangeFactorization f) := by\n  intro X Y f\n  exact (Set.rangeFactorization_surjective)\nexact @key (Fin 2) (Fin 2) (id)'),
    (['ℝ', 'ℝ', 'Real.sin'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Surjective (Set.rangeFactorization f) := by\n  intro X Y f\n  exact (Set.rangeFactorization_surjective)\nexact @key (ℝ) (ℝ) (Real.sin)'),
    (['ℝ', 'ℝ', 'fun x => x ^ 2 + 1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Surjective (Set.rangeFactorization f) := by\n  intro X Y f\n  exact (Set.rangeFactorization_surjective)\nexact @key (ℝ) (ℝ) (fun x => x ^ 2 + 1)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
