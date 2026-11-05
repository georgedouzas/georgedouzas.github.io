# name: set-three-maps-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Injective.three_maps_injectivity
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['Fin 4', 'inferInstance', '0', '1', '2', '3', '(by decide)', '(by decide)', '(by decide)', '(by decide)', '(by decide)', '(by decide)'], True, 'decide'),
    (['ℕ', 'inferInstance', '10', '20', '30', '40', '(by decide)', '(by decide)', '(by decide)', '(by decide)', '(by decide)', '(by decide)'], False, 'set_option maxHeartbeats 2000000 in\nhave key : ∀ {β : Type} [inst1 : DecidableEq β] (a b c d : β) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d), Function.Injective ![a, b, c] ∧ ¬ Function.Injective ![a, a, b] ∧ Function.Injective ![d, c, b] := by\n  intro β inst1 a b c d hab hac had hbc hbd hcd\n  refine ⟨?_, fun h => ?_, ?_⟩\n  · intro i j h\n    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]\n  · exact absurd (h (show ![a, a, b] 0 = ![a, a, b] 1 from rfl)) (by decide)\n  · intro i j h\n    fin_cases i <;> fin_cases j <;> simp_all [eq_comm]\nexact @key (ℕ) (inferInstance) (10) (20) (30) (40) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide)) ((by decide))'),
    (['Fin 5', 'inferInstance', '4', '2', '0', '1', '(by decide)', '(by decide)', '(by decide)', '(by decide)', '(by decide)', '(by decide)'], False, 'decide'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
