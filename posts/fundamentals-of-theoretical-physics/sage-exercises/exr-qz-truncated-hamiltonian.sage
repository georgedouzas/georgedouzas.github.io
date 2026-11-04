# Exercise exr-qz-truncated-hamiltonian: building q and p from truncated ladder operators (N = 6) and forming
# (q^2 + p^2)/2 spoils the last level: the diagonal is n + 1/2 for n < N - 1 and (N - 1)/2 at the end.
N = 6
a = matrix(SR, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0); ad = a.transpose()
q = (a + ad) / sqrt(2); p = -I * (a - ad) / sqrt(2)
H1 = ((q * q + p * p) / 2).apply_map(lambda e: e.simplify_full())
print("(q^2 + p^2)/2 is diagonal:", H1.is_diagonal(), " entries:", H1.diagonal())
print("equals (a a^dag + a^dag a)/2:", H1 == ((a * ad + ad * a) / 2).apply_map(lambda e: e.simplify_full()))
H2 = ad * a + identity_matrix(SR, N) / 2
print("a^dag a + 1/2 has entries:", H2.diagonal(), " (exact n + 1/2)")
