# Exercise exr-qm-coherent-truncated: a coherent state with alpha = 1 + i in a truncated 30-dimensional basis.
N = 30; al = CDF(1, 1)
v = vector(CDF, [exp(-abs(al) ** 2 / 2) * al ** n / sqrt(factorial(n)) for n in range(N)])
a = matrix(CDF, N, N, lambda i, j: sqrt(j) if i == j - 1 else 0)
print("||v|| - 1 =", v.norm() - 1, "; ||a v - alpha v|| =", (a * v - al * v).norm())
X = (a + a.conjugate_transpose()) / sqrt(2.0)
for t in [0, 0.5, 1, 2]:
    vt = vector(CDF, [exp(-I * (n + 0.5) * t) * c for n, c in enumerate(v)])
    mx = (vt.conjugate() * X * vt).real()
    print("t =", t, " <x> =", round(mx, 12), " sqrt 2 |alpha| cos(t - pi/4) =", round(sqrt(2.0) * abs(al) * cos(t - pi / 4).n(), 12))
