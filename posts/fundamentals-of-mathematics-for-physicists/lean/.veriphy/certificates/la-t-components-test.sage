# name: la-t-components-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.MultilinearMaps.T_components file "la-t-components-test.sage"
# description: the 27 components of T(u, v, w) = u¹ v² w³ − u² v¹ w³ at every index triple (exhaustive), by evaluation.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [T, Pi.single_apply]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0', '0', '0'], True),
    (['0', '0', '1'], False),
    (['0', '0', '2'], False),
    (['0', '1', '0'], False),
    (['0', '1', '1'], False),
    (['0', '1', '2'], False),
    (['0', '2', '0'], False),
    (['0', '2', '1'], False),
    (['0', '2', '2'], False),
    (['1', '0', '0'], False),
    (['1', '0', '1'], False),
    (['1', '0', '2'], False),
    (['1', '1', '0'], False),
    (['1', '1', '1'], False),
    (['1', '1', '2'], False),
    (['1', '2', '0'], False),
    (['1', '2', '1'], False),
    (['1', '2', '2'], False),
    (['2', '0', '0'], False),
    (['2', '0', '1'], False),
    (['2', '0', '2'], False),
    (['2', '1', '0'], False),
    (['2', '1', '1'], False),
    (['2', '1', '2'], False),
    (['2', '2', '0'], False),
    (['2', '2', '1'], False),
    (['2', '2', '2'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
