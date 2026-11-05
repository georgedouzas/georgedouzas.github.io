# name: la-functions-infinite-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.not_finiteDimensional_functions file "la-functions-infinite-test.sage"
# description: the functions on ℕ, ℤ and ℚ are infinite-dimensional: the indicator functions are independent and infinite.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['ℕ', 'inferInstance'], True, 'intro h\nclassical\nhave := (Pi.linearIndependent_single_one ℕ ℝ).finite_of_isNoetherian\nexact not_finite ℕ'),
    (['ℤ', 'inferInstance'], False, 'intro h\nclassical\nhave := (Pi.linearIndependent_single_one ℤ ℝ).finite_of_isNoetherian\nexact not_finite ℤ'),
    (['ℚ', 'inferInstance'], False, 'intro h\nclassical\nhave := (Pi.linearIndependent_single_one ℚ ℝ).finite_of_isNoetherian\nexact not_finite ℚ'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
