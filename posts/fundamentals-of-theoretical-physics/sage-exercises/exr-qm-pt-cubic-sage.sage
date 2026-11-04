# Exercise exr-qm-pt-cubic-sage: second-order shift of the ground state for g x^3 (oscillator units).
N = 10
a = matrix(SR, N, N, lambda i, j: sqrt(j) if i == j - 1 else 0); X = (a + a.transpose()) / sqrt(2)
X3 = (X ** 3).simplify_full()
print("diagonal of x^3:", X3.diagonal()[:N - 3])
print("x^3 |0> =", X3.column(0))
g = var("g")
print("second-order shift of E_0 =", (g ** 2 * sum(X3[k, 0] ** 2 / (0 - k) for k in range(1, N))).simplify_full())
NB = 60; an = matrix(RDF, NB, NB, lambda i, j: sqrt(j) if i == j - 1 else 0); Xn = (an + an.transpose()) / sqrt(2.0)
gg = 0.02
E0 = min((diagonal_matrix(RDF, [n + 0.5 for n in range(NB)]) + gg * Xn ** 3).eigenvalues(), key=lambda z: abs(z - 0.5))
print("g = 0.02: numerical E_0 =", E0, " second order 1/2 - 11 g^2/8 =", 0.5 - 11 / 8 * gg ** 2)
