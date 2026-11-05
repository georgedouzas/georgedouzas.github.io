# name: la-rn-axioms-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.rn_axioms file "la-rn-axioms-test.sage"
# description: the four scalar rules in ℝⁿ at n = 0, 2, 3 with explicit scalars and vectors, entry by entry.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'refine ⟨?_, ?_, ?_, ?_⟩ <;> funext i <;> fin_cases i <;> norm_num'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0', '![]', '![]'], True),
    (['2', '2', '-1', '![1, 2]', '![3, -1]'], False),
    (['3', '1 / 2', '3', '![1, 0, -2]', '![0, 5, 1]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
