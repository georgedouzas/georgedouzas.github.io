# Exercise exr-ccr-truncated-ladder: x, p and H = (x^2 + p^2)/2 in the truncated ladder basis (N = 6), the eigenvalues of
# H against n + 1/2, and the conjugation e^{i theta a^dagger a} a e^{-i theta a^dagger a} = e^{-i theta} a.
N = 6
a = matrix(SR, N, N, lambda j, k: sqrt(k) if k == j + 1 else 0)
ad = a.conjugate_transpose()
x = (a + ad) / sqrt(2); p = I * (ad - a) / sqrt(2)
H = ((x * x + p * p) / 2).simplify_full()
print("H = (x^2 + p^2)/2 ="); print(H)
print("eigenvalues of H:", sorted(H.eigenvalues()), "; exact n + 1/2:", [n + QQ(1) / 2 for n in range(N)])
print("H = a^dagger a + [a, a^dagger]/2:", bool(H == ad * a + (a * ad - ad * a) / 2))
th = var("theta", domain="real")
Nop = ad * a
U = diagonal_matrix([exp(I * th * Nop[k, k]) for k in range(N)])
print("e^{i theta N} a e^{-i theta N} = e^{-i theta} a:", bool((U * a * U.conjugate_transpose() - exp(-I * th) * a).simplify_full() == 0))
