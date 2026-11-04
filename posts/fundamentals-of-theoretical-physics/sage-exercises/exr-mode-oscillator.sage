# Exercise exr-mode-oscillator: the harmonic oscillator quantised with ladder operators, represented as truncated
# matrices in Sage: the commutator [a, a^dagger] = 1, the number operator, the spectrum (n + 1/2) omega, and the
# position-momentum commutator [q, p] = i with q = (a + a^dagger)/sqrt(2), p = -i(a - a^dagger)/sqrt(2).
N = 8
a = matrix(SR, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0)
ad = a.transpose()
num = ad * a
print("number operator a^dagger a is diagonal with entries 0..N-1:", num == diagonal_matrix(SR, [n for n in range(N)]))
comm = a * ad - ad * a
print("[a, a^dagger] = 1 except for the truncation entry:", [comm[i, i] for i in range(N)])
q = (a + ad) / sqrt(2); p = -I * (a - ad) / sqrt(2)
qp = q * p - p * q
print("[q, p] = i (off the truncation):", [qp[i, i] for i in range(N - 1)])
w = var("w", domain="real")
H = w * (num + identity_matrix(SR, N) / 2)
print("H = w (a^dagger a + 1/2) has eigenvalues", [H[n, n] for n in range(4)], "...")
print("a lowers and a^dagger raises the level: a|2> = sqrt(2)|1>:", (a * vector(SR, [0, 0, 1] + [0] * (N - 3))) == sqrt(2) * vector(SR, [0, 1] + [0] * (N - 2)))
