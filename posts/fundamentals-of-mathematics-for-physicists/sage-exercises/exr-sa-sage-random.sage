# Exercise exr-sa-sage-random: is_hermitian for a tridiagonal complex matrix, reality of <u, M u> for a symbolic u, and the
# Hermitian and anti-Hermitian parts of a non-Hermitian matrix.
M = matrix(SR, [[2, I, 0], [-I, 2, I], [0, -I, 2]])
print("M is Hermitian:", M.is_hermitian())
u = vector(SR, var("u1 u2 u3"))
q = sum(conjugate(a) * b for a, b in zip(u, M * u)).expand()
print("<u, M u> =", q, "; imaginary part:", (q - conjugate(q)).simplify_full() / (2 * I))
N = matrix(SR, [[1, 2], [3, 4 * I]])
B = (N + N.conjugate_transpose()) / 2; C = (N - N.conjugate_transpose()) / (2 * I)
print("B ="); print(B); print("C ="); print(C)
print("B, C Hermitian:", B.is_hermitian(), C.is_hermitian(), "; B + iC = N:", bool(B + I * C == N))
