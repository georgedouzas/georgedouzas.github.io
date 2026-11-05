# name: set-pair-injective-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.pair_injective
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 3', '0', '1', '(by decide)'], True, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Function.Injective ![x, y] := by\n  intro β x y hxy\n  intro i j h\n  fin_cases i <;> fin_cases j <;> simp_all [eq_comm]\nexact @key (Fin 3) (0) (1) ((by decide))'),
    (['ℕ', '4', '9', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Function.Injective ![x, y] := by\n  intro β x y hxy\n  intro i j h\n  fin_cases i <;> fin_cases j <;> simp_all [eq_comm]\nexact @key (ℕ) (4) (9) ((by decide))'),
    (['Bool', 'true', 'false', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} (x y : β) (hxy : x ≠ y), Function.Injective ![x, y] := by\n  intro β x y hxy\n  intro i j h\n  fin_cases i <;> fin_cases j <;> simp_all [eq_comm]\nexact @key (Bool) (true) (false) ((by decide))'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
