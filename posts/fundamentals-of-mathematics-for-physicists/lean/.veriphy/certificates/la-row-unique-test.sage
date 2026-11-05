# name: la-row-unique-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.row_unique file "la-row-unique-test.sage"
# description: two rows giving the same functional are equal, at n = 0, 2, 3 (the hypothesis forces equal rows).
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'rfl'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '![]', '![]', '(fun _ => rfl)'], True),
    (['2', '![2, 1]', '![2, 1]', '(fun _ => rfl)'], False),
    (['3', '![3, -1, 2]', '![3, -1, 2]', '(fun _ => rfl)'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
