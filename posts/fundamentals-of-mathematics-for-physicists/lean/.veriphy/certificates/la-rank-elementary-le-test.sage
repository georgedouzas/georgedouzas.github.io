# name: la-rank-elementary-le-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.ElementaryTensors.rank_elementary_le file "la-rank-elementary-le-test.sage"
# description: an elementary tensor of ℚ³ ⊗ ℚ³ has rank at most 1: Sage writes u vᵀ as a 3 × 1 times 1 × 3 product, Lean checks it.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0'], True, 'exact Veriphy.Linear.rank_le_of_factor _ !![0; 0; 0] !![0, 0, 0] (by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.vecMulVec_apply, Matrix.mul_apply])'),
    (['![1, 2, 3]', '![1, 0, -1]'], False, 'exact Veriphy.Linear.rank_le_of_factor _ !![1; 2; 3] !![1, 0, -1] (by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.vecMulVec_apply, Matrix.mul_apply])'),
    (['![2, -1, 0]', '![0, 0, 5]'], False, 'exact Veriphy.Linear.rank_le_of_factor _ !![2; -1; 0] !![0, 0, 5] (by ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.vecMulVec_apply, Matrix.mul_apply])'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
