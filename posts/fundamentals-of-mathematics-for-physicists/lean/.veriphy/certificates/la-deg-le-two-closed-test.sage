# name: la-deg-le-two-closed-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.deg_le_two_closed file "la-deg-le-two-closed-test.sage"
# description: sums and multiples of polynomials of degree at most 2 stay of degree at most 2, at explicit polynomials.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'constructor\n· rw [Polynomial.mem_degreeLT]\n  refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))\n  first | compute_degree! | simp\n· rw [Polynomial.mem_degreeLT]\n  refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3))\n  first | compute_degree! | simp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '(by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)', '(by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)'], True),
    (['Polynomial.X ^ 2 + 1', '3 * Polynomial.X', '2', '(by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)', '(by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)'], False),
    (['Polynomial.X ^ 2', '-Polynomial.X ^ 2 + Polynomial.X', '-1 / 2', '(by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)', '(by rw [Polynomial.mem_degreeLT]; refine lt_of_le_of_lt (b := ((2 : ℕ) : WithBot ℕ)) ?_ (by exact_mod_cast (by norm_num : 2 < 3)); first | compute_degree! | simp)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
