# name: set-samevalue-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Equivalence.sameValue_equivalence
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['ℝ', 'ℝ', 'fun x => x ^ 2'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Equivalence (fun x x' => f x = f x') := by\n  intro X Y f\n  exact (⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩)\nexact @key (ℝ) (ℝ) (fun x => x ^ 2)"),
    (['Unit', 'Unit', 'id'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Equivalence (fun x x' => f x = f x') := by\n  intro X Y f\n  exact (⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩)\nexact @key (Unit) (Unit) (id)"),
    (['Matrix (Fin 3) (Fin 3) ℝ', 'Fin 3 → ℝ', 'KaluzaKlein.Sets.Maps.orbit'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ {X Y : Type} (f : X → Y), Equivalence (fun x x' => f x = f x') := by\n  intro X Y f\n  exact (⟨fun _ => rfl, fun h => h.symm, fun h1 h2 => h1.trans h2⟩)\nexact @key (Matrix (Fin 3) (Fin 3) ℝ) (Fin 3 → ℝ) (KaluzaKlein.Sets.Maps.orbit)"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
