# name: la-rank-one-iff-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.DualResults.rank_one_iff_proportional file "la-rank-one-iff-test.sage"
# description: a 2 × n matrix with non-zero first row has rank 1 exactly when its rows are proportional: Sage compares the rank with the proportionality of the rows at matrices with a zero, a proportional and an independent second row.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
MATRICES = [([[1, 2], [0, 0]], True), ([[1, 2, 3], [2, 4, 6]], False), ([[1, 0, 2], [0, 1, 1]], False),
            ([[2, -1], [-4, 2]], False)]  # EDIT: the instances (the zero second row is the edge case)
lean = lambda m: '!![' + '; '.join(', '.join(str(x) for x in r) for r in m) + ']'
instances = []
for rows, edge in MATRICES:
    a = matrix(QQ, rows)
    rank_one = Integer(a.rank() == 1)
    proportional = Integer(matrix(QQ, [a.row(0), a.row(1)]).rank() <= 1)  # A 1 = c • A 0 for some c, as A 0 ≠ 0
    instances.append(oracle_instance('ℚ', 'inferInstance', str(a.ncols()), lean(rows), '(by decide)',
                                     lhs=rank_one, rhs=proportional, edge=edge))
