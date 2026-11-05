# name: set-projections-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Products.projections_surjective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', 'Fin 3', 'inferInstance', 'inferInstance'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : Nonempty X] [inst2 : Nonempty Y], Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) := by\n  intro X Y inst1 inst2\n  exact (⟨Prod.fst_surjective, Prod.snd_surjective⟩)\nexact @key (Fin 2) (Fin 3) (inferInstance) (inferInstance)'),
    (['Unit', 'Unit', 'inferInstance', 'inferInstance'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : Nonempty X] [inst2 : Nonempty Y], Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) := by\n  intro X Y inst1 inst2\n  exact (⟨Prod.fst_surjective, Prod.snd_surjective⟩)\nexact @key (Unit) (Unit) (inferInstance) (inferInstance)'),
    (['ℝ', 'ℕ', 'inferInstance', 'inferInstance'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ (X Y : Type) [inst1 : Nonempty X] [inst2 : Nonempty Y], Function.Surjective (Prod.fst : X × Y → X) ∧ Function.Surjective (Prod.snd : X × Y → Y) := by\n  intro X Y inst1 inst2\n  exact (⟨Prod.fst_surjective, Prod.snd_surjective⟩)\nexact @key (ℝ) (ℕ) (inferInstance) (inferInstance)'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
