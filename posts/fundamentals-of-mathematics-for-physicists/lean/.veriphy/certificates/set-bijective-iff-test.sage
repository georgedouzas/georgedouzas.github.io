# name: set-bijective-iff-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Bijective.bijective_iff_existsUnique
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 2', 'Fin 2', 'id'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∀ y, ∃! x, f x = y := by\n  intro X Y f\n  exact (Function.bijective_iff_existsUnique f)\nexact @key (Fin 2) (Fin 2) (id)'),
    (['ℤ', 'ℤ', 'fun k => k + 1'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∀ y, ∃! x, f x = y := by\n  intro X Y f\n  exact (Function.bijective_iff_existsUnique f)\nexact @key (ℤ) (ℤ) (fun k => k + 1)'),
    (['Fin 3', 'Fin 2', '![0, 0, 1]'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Function.Bijective f ↔ ∀ y, ∃! x, f x = y := by\n  intro X Y f\n  exact (Function.bijective_iff_existsUnique f)\nexact @key (Fin 3) (Fin 2) (![0, 0, 1])'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
