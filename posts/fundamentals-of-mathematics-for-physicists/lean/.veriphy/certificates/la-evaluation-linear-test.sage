# name: la-evaluation-linear-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.LinearFunctionals.evaluation_linear file "la-evaluation-linear-test.sage"
# description: evaluation at 0 is linear, at the zero functions, polynomial and trigonometric functions, by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0'], True),
    (['(fun x => x ^ 2 + 1)', 'Real.cos', '3'], False),
    (['Real.exp', '(fun x => 2 * x - 5)', '-1 / 2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
