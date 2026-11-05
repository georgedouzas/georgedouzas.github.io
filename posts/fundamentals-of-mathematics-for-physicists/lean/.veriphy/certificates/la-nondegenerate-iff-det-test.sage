# name: la-nondegenerate-iff-det-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.NondegenerateForms.nondegenerate_iff_det file "la-nondegenerate-iff-det-test.sage"
# description: B is non-degenerate exactly when det B ≠ 0: Sage compares the triviality of the kernel with det B ≠ 0 at the zero form, the degenerate S, the Euclidean form and the Minkowski form.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
MATRICES = [([[0, 0], [0, 0]], True), ([[1, 2], [2, 4]], False), ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], False),
            ([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], False)]  # EDIT: the instances
lean = lambda m: '!![' + '; '.join(', '.join(str(x) for x in r) for r in m) + ']'
instances = []
for rows, edge in MATRICES:
    b = matrix(QQ, rows)
    nondegenerate = Integer(b.right_kernel().dimension() == 0)  # B v = 0 forces v = 0
    instances.append(oracle_instance(str(b.nrows()), lean(rows), lhs=nondegenerate, rhs=Integer(b.det() != 0),
                                     edge=edge))
