# name: la-dual-expansion-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualSpace.dual_expansion file "la-dual-expansion-test.sage"
# description: α = α(e_i) e^i for the zero functional and coordinate functionals of ℚ², checked on every vector.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'ext v\nsimp [Fin.sum_univ_succ, Basis.dualBasis_apply, Pi.basisFun_repr]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℚ', '(Fin 2 → ℚ)', '(Fin 2)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 2)', '0'], True),
    (['ℚ', '(Fin 2 → ℚ)', '(Fin 2)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 2)', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 2 => ℚ) 1'], False),
    (['ℚ', '(Fin 3 → ℚ)', '(Fin 3)', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'inferInstance', 'Pi.basisFun ℚ (Fin 3)', 'LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 0'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
