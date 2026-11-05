# name: la-functional-is-row-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.functional_is_row file "la-functional-is-row-test.sage"
# description: α(v) = Σ v^i α(e_i) for the zero functional, a coordinate and a row functional, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [Fin.sum_univ_succ, Matrix.mulVec, dotProduct]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['2', '0', '![1, 2]'], True),
    (['2', 'LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℝ) 1', '![1, 2]'], False),
    (['3', "LinearMap.proj (R := ℝ) (φ := fun _ : Fin 1 => ℝ) 0 ∘ₗ Matrix.toLin' !![3, -1, 2]", '![1, 0, -2]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
