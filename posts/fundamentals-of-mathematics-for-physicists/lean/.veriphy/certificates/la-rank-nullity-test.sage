# name: la-rank-nullity-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearMaps.rank_nullity file "la-rank-nullity-test.sage"
# description: dim ker L + dim L(V) = dim V for the coordinate functionals of ℚ¹, ℚ² and ℚ³ (their kernels by Veriphy.Linear.finrank_ker_add_one, their images all of ℚ).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 1 → ℚ)', 'ℚ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 1 => ℚ) 0'], True, 'have hk := Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 1 => ℚ) 0) ![1] (by simp)\nhave hr : LinearMap.range (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 1 => ℚ) 0) = ⊤ :=\n  LinearMap.range_eq_top.mpr fun c => ⟨Pi.single 0 c, by simp⟩\nrw [hr, finrank_top]\nsimp at hk ⊢\nomega'),
    (['ℚ', '(Fin 2 → ℚ)', 'ℚ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1'], False, 'have hk := Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) ![0, 1] (by simp)\nhave hr : LinearMap.range (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1) = ⊤ :=\n  LinearMap.range_eq_top.mpr fun c => ⟨Pi.single 1 c, by simp⟩\nrw [hr, finrank_top]\nsimp at hk ⊢\nomega'),
    (['ℚ', '(Fin 3 → ℚ)', 'ℚ', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0'], False, 'have hk := Veriphy.Linear.finrank_ker_add_one (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) ![1, 0, 0] (by simp)\nhave hr : LinearMap.range (LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0) = ⊤ :=\n  LinearMap.range_eq_top.mpr fun c => ⟨Pi.single 0 c, by simp⟩\nrw [hr, finrank_top]\nsimp at hk ⊢\nomega'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
