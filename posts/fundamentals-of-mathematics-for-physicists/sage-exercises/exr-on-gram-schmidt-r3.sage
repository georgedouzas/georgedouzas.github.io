# Exercise exr-on-gram-schmidt-r3: Gram-Schmidt on (1, 1, 0), (1, 0, 1), (0, 1, 1) in R^3, the orthonormal frame, and
# the check that it spans the same space.
V = matrix(QQ, [[1, 1, 0], [1, 0, 1], [0, 1, 1]])
W, mu = V.gram_schmidt()
print("orthogonal vectors w_k (rows):"); print(W)
E = matrix(SR, [w / w.norm() for w in W.rows()])
print("orthonormal frame e_k (rows):"); print(E)
print("Gram matrix E E^T = 1:", bool((E * E.transpose()).simplify_full() == identity_matrix(3)))
print("same span as the v_k:", V.row_space() == W.row_space())
