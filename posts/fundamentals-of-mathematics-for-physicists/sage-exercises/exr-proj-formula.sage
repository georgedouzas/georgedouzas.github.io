# Exercise exr-proj-formula: the matrix P = E E^dagger of the orthogonal projection onto span{(1, 1, 0), (0, 1, i)} in C^3,
# the equations P^2 = P = P^dagger, its eigenvalues and trace.
V = matrix(SR, [[1, 1, 0], [0, 1, I]])
# Gram-Schmidt by hand in the complex inner product (antilinear in the first slot)
ip = lambda a, b: sum(conjugate(x) * y for x, y in zip(a, b))
w1 = V.row(0); w2 = V.row(1) - ip(w1, V.row(1)) / ip(w1, w1) * w1
e1 = w1 / sqrt(ip(w1, w1)); e2 = w2 / sqrt(ip(w2, w2))
E = matrix(SR, [e1, e2]).transpose()
P = (E * E.conjugate_transpose()).simplify_full()
print("P ="); print(P)
print("P^2 = P:", bool((P * P).simplify_full() == P), "; P^dagger = P:", P.is_hermitian())
print("eigenvalues:", P.eigenvalues(), "; trace:", P.trace().simplify_full())
