# Exercise exr-qm-cg-sage: the Clebsch-Gordan table of 1 x 1/2, its orthogonality, and the scalar state |0, 0> of 1 x 1.
from sage.functions.wigner import clebsch_gordan
j1, j2 = 1, 1 / 2
rows = [(m1, m2) for m1 in [1, 0, -1] for m2 in [1 / 2, -1 / 2]]
cols = [(3 / 2, m) for m in [3 / 2, 1 / 2, -1 / 2, -3 / 2]] + [(1 / 2, m) for m in [1 / 2, -1 / 2]]
C = matrix(SR, 6, 6, lambda r, c: SR(clebsch_gordan(j1, j2, cols[c][0], rows[r][0], rows[r][1], cols[c][1])).simplify_full())
print("rows |m1, m2> =", rows); print("columns |j, m> =", cols); print(C)
print("C^T C = 1:", (C.transpose() * C).simplify_full() == identity_matrix(6))
print("|0,0> in 1 x 1:", [(m1, -m1, SR(clebsch_gordan(1, 1, 0, m1, -m1, 0)).simplify_full()) for m1 in [1, 0, -1]])
