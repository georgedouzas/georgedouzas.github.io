# Exercise exr-qm-ladder-truncated-sage: truncated N = 8 ladder matrices, the truncated H, and the eigenvalues of x.
N = 8
a = matrix(SR, N, N, lambda i, j: sqrt(j) if i == j - 1 else 0); ad = a.transpose()
print("[a, a^dagger] = diag", (a * ad - ad * a).diagonal())
X = (a + ad) / sqrt(2); P = I * (ad - a) / sqrt(2)
H = ((P * P + X * X) / 2).apply_map(lambda e: e.simplify_full())
print("(p^2 + x^2)/2 - diag(n + 1/2) =", (H - diagonal_matrix(SR, [n + 1 / 2 for n in range(N)])).diagonal(), "(off-diagonal zero:", (H - diagonal_matrix(H.diagonal())).is_zero(), ")")
ev = sorted(matrix(RDF, X).eigenvalues())
xi = polygen(RDF, "xi")
zeros = sorted(r for r, _ in hermite(N, xi).roots())
print("eigenvalues of the truncated x:", [round(e, 6) for e in ev])
print("zeros of H_8:                  ", [round(z, 6) for z in zeros])
print("max difference:", max(abs(e - z) for e, z in zip(ev, zeros)))
