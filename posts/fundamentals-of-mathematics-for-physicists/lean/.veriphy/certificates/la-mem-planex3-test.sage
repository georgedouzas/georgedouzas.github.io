# name: la-mem-planex3-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.VectorSpaces.mem_planeX3 file "la-mem-planex3-test.sage"
# description: membership in the plane x³ = 0 at the origin, at a point of the plane and at a point off it.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = 'simp [planeX3]'  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True),
    (['![1, 2, 0]'], False),
    (['![1, 2, 3]'], False),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
