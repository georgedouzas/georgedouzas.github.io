# Exercise exr-adj-sage-check: the adjoint identity <A u, v> = <u, A^dagger v> as a polynomial identity in the components
# of symbolic vectors u, v in C^3, its failure for the plain transpose, and (AB)^dagger = B^dagger A^dagger.
A = matrix(SR, [[1, I, 0], [2, 0, 1 - I], [0, 3 * I, 1]])
u = vector(SR, var("u1 u2 u3")); v = vector(SR, var("v1 v2 v3"))
ip = lambda a, b: sum(conjugate(x) * y for x, y in zip(a, b))
print("A^dagger ="); print(A.conjugate_transpose())
print("<A u, v> - <u, A^dagger v> =", (ip(A * u, v) - ip(u, A.conjugate_transpose() * v)).simplify_full())
print("<A u, v> - <u, A^T v> =", (ip(A * u, v) - ip(u, A.transpose() * v)).simplify_full())
B = A ** 2 + I * A
print("(AB)^dagger = B^dagger A^dagger:", bool((A * B).conjugate_transpose() == B.conjugate_transpose() * A.conjugate_transpose()))
