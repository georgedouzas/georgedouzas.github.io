# name: ra-jacobian-linear-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.jacobian_linear file "ra-jacobian-linear-test.sage"
# description: the derivative of x ↦ Ax is A, of x ↦ x + c the identity (explicit A, c, p)
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['1', '0', '0'], True, "refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' (1 : Matrix (Fin 2) (Fin 2) ℝ))).hasFDerivAt, ?_⟩\nsimpa using (hasFDerivAt_id (𝕜 := ℝ) (0 : Fin 2 → ℝ)).add_const (0 : Fin 2 → ℝ)"), (['!![(1 : ℝ), 2; 3, 4]', '![1, -1]', '![0, 5]'], False, "refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' (!![(1 : ℝ), 2; 3, 4] : Matrix (Fin 2) (Fin 2) ℝ))).hasFDerivAt, ?_⟩\nsimpa using (hasFDerivAt_id (𝕜 := ℝ) (![0, 5] : Fin 2 → ℝ)).add_const (![1, -1] : Fin 2 → ℝ)"), (['!![(0 : ℝ), -1; 1, 0]', '![2, 2]', '![1, 1]'], False, "refine ⟨(LinearMap.toContinuousLinearMap (𝕜 := ℝ) (E := Fin 2 → ℝ) (F' := Fin 2 → ℝ) (Matrix.toLin' (!![(0 : ℝ), -1; 1, 0] : Matrix (Fin 2) (Fin 2) ℝ))).hasFDerivAt, ?_⟩\nsimpa using (hasFDerivAt_id (𝕜 := ℝ) (![1, 1] : Fin 2 → ℝ)).add_const (![2, 2] : Fin 2 → ℝ)")]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
