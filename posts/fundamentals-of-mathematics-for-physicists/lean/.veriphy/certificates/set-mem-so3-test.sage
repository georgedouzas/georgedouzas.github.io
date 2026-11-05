# name: set-mem-so3-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Sets.mem_SO3_iff
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1'], True, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (R : Matrix (Fin 3) (Fin 3) ℝ), R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by\n  intro R\n  rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']\nexact @key (1)"),
    (['!![1, 0, 0; 0, -1, 0; 0, 0, -1]'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (R : Matrix (Fin 3) (Fin 3) ℝ), R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by\n  intro R\n  rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']\nexact @key (!![1, 0, 0; 0, -1, 0; 0, 0, -1])"),
    (['!![2, 0, 0; 0, 1, 0; 0, 0, 1]'], False, "set_option maxHeartbeats 2000000 in\nhave key : ∀ (R : Matrix (Fin 3) (Fin 3) ℝ), R ∈ SO3 ↔ Rᵀ * R = 1 ∧ R.det = 1 := by\n  intro R\n  rw [SetLike.mem_coe, Matrix.mem_specialOrthogonalGroup_iff, Matrix.mem_orthogonalGroup_iff']\nexact @key (!![2, 0, 0; 0, 1, 0; 0, 0, 1])"),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
