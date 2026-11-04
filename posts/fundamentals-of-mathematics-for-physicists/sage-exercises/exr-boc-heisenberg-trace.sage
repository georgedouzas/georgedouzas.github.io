# Exercise exr-boc-heisenberg-trace: truncated ladder operators a (a_{n, n+1} = sqrt n) in dimension N = 5, the commutator
# [a, a^dagger] = diag(1, ..., 1, 1 - N) of trace zero, tr[A, B] = 0 for symbolic 3 x 3 matrices, and [x, p].
N = 5
a = matrix(SR, N, N, lambda j, k: sqrt(k) if k == j + 1 else 0)
ad = a.conjugate_transpose()
C = a * ad - ad * a
print("[a, a^dagger] ="); print(C)
print("trace:", C.trace())
X = matrix(SR, 3, 3, var(" ".join("x%d%d" % (j, k) for j in range(3) for k in range(3))))
Y = matrix(SR, 3, 3, var(" ".join("y%d%d" % (j, k) for j in range(3) for k in range(3))))
print("tr[A, B] for symbolic 3 x 3 matrices:", (X * Y - Y * X).trace().expand())
x = (a + ad) / sqrt(2); p = I * (ad - a) / sqrt(2)
print("x, p Hermitian:", x.is_hermitian(), p.is_hermitian(), "; [x, p] = i [a, a^dagger]:", bool((x * p - p * x).simplify_full() == I * C))
