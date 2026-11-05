# name: la-det-signs-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MinkowskiResults.det_signs file "la-det-signs-test.sage"
# description: det diag(±1) = (−1)^q, q the number of −1 entries, at n = 0 and sign patterns of length 2, 4, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]'], True, 'simp'),
    (['2', '![true, false]'], False, 'rw [Matrix.det_diagonal, show (Finset.univ.filter fun i => (![true, false]) i = true).card = 1 from by decide]\nnorm_num [Fin.prod_univ_succ]'),
    (['4', '![false, true, true, true]'], False, 'rw [Matrix.det_diagonal, show (Finset.univ.filter fun i => (![false, true, true, true]) i = true).card = 3 from by decide]\nnorm_num [Fin.prod_univ_succ]'),
    (['3', '![true, true, false]'], False, 'rw [Matrix.det_diagonal, show (Finset.univ.filter fun i => (![true, true, false]) i = true).card = 2 from by decide]\nnorm_num [Fin.prod_univ_succ]'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
